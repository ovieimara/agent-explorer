import argparse
import datetime
import playwright
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from playwright.async_api import async_playwright
import asyncio
from rag import get_embedding_model, create_rag_chain, create_or_load_vector_store, DEEPSEEK_MODEL, GEMMA_MODEL, MISTRAL_MODEL,  EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH


class WebFormAgent:
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain
        # Add field type to question mapping
        self.field_type_questions = {
            'email': "What is the email address?",
            'password': "What is the password?",
            'tel': "What is the phone number?",
            'text': "What text should be entered here?",
            'textarea': "What content should be written?",
            'select': "Which option should be selected?"
        }

    def get_field_value(self, field_info):
        # Get question text with validation
        question = field_info.get('placeholder') or \
            field_info.get('name') or \
            field_info.get('id') or \
            self.field_type_questions.get(field_info.get(
                'type'), "What value should be entered here?")

        # Ensure we return a clean string
        return str(question).strip() if question else "What value should be entered here?"

    def _get_clean_answer(self, rag_response):
        """Extract and validate answer from RAG response structure"""
        try:
            # Handle AIMessage format
            if hasattr(rag_response['answer'], 'content'):
                answer = rag_response['answer'].content
            else:
                answer = rag_response['answer']

            # Clean and validate
            answer = str(answer).strip()
            if not answer or answer.lower() == "there is no value to extract":
                return None

            return answer

        except (KeyError, AttributeError) as e:
            print(f"Invalid RAG response format: {str(e)}")
            return None

    def set_fields(self, current_fields):
        field_data = []
        for field in current_fields:
            try:
                question = self.get_field_value(field)
                if not question:
                    print(f"Skipping field with no question: {field}")
                    continue

                rag_response = self.rag_chain.invoke(question)
                answer = self._get_clean_answer(rag_response)

                if answer:
                    field_data.append({**field, 'value': answer})
                else:
                    print(f"No valid answer for: {question}")

            except Exception as e:
                print(f"Error processing field {field}: {str(e)}")

        return field_data

    def _get_best_selector(self, field):
        """Priority-based selector construction"""
        if field['id']:
            return f'#{field["id"]}'
        if field['name']:
            return f'[name="{field["name"]}"]'
        if field['aria-label']:
            return f'[aria-label="{field["aria-label"]}"]'
        return None

    async def _locate_current_fields(self, page):
        """Find all visible form fields on current page (excluding submit buttons)"""
        fields = []
        selectors = [
            'input:not([type="submit"]):visible',
            'textarea:visible',
            'select:visible',
            '[role="textbox"]:visible',
            '[contenteditable="true"]:visible'
        ]
        for selector in selectors:
            elements = await page.query_selector_all(selector)
            for el in elements:
                if await el.is_visible():
                    field_info = await self._get_field_info(el)
                    # Filter out submit buttons
                    if field_info.get('type') != 'submit':
                        fields.append(field_info)
        return fields

    async def _get_field_info(self, element):
        """Extract metadata from form element"""
        return {
            'tag': await element.evaluate('el => el.tagName.toLowerCase()'),
            'type': await element.get_attribute('type'),
            'name': await element.get_attribute('name'),
            'id': await element.get_attribute('id'),
            'placeholder': await element.get_attribute('placeholder'),
            'aria-label': await element.get_attribute('aria-label'),
            'source': 'playwright'
        }

    async def fill_form_with_playwright(self, url):
        """Handle both single and multi-page forms with user confirmation"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,
                args=["--start-maximized"]
            )
            context = await browser.new_context(
                no_viewport=True,
                ignore_https_errors=True
            )
            page = await context.new_page()

            try:
                await page.goto(url, timeout=15000)
                screen = await page.evaluate("""() => ({
                    width: window.screen.availWidth,
                    height: window.screen.availHeight
                })""")
                await page.set_viewport_size(screen)

                page_num = 1
                while True:
                    print(f"\nProcessing page {page_num}")

                    # Check for form existence
                    form = await page.query_selector("form:not(.search-form)")
                    if not form:
                        print("No form found - ending form fill process")
                        break

                    # Locate fields on current page
                    current_fields = await self._locate_current_fields(page)
                    if not current_fields:
                        print("No fields found - ending form fill process")
                        break

                    # Generate answers and fill fields
                    field_data = self.set_fields(current_fields)
                    for field in field_data:
                        selector = self._get_best_selector(field)
                        if selector:
                            await page.fill(selector, field['value'])
                        else:
                            print(f"Skipping field with no selector: {field}")

                    # Handle form submission with user confirmation
                    submit_selector = ('input[type="submit"], '
                                       'button[type="submit"], '
                                       'button:has-text("Next"), '
                                       'button:has-text("Continue")')
                    submit_button = await page.query_selector(submit_selector)

                    if submit_button:
                        # Get button type and text for verification
                        button_type = await submit_button.get_attribute('type')
                        button_text = await submit_button.inner_text()

                        # Only prompt for explicit submit buttons
                        if button_type == 'submit' or 'submit' in button_text.lower():
                            print(f"\nFound submit button: {button_text}")
                            answer = await asyncio.to_thread(
                                lambda: input(
                                    "Proceed with submission? (Y/n) ").strip().lower()
                            )
                            if answer not in ('', 'y', 'yes'):
                                print("Submission cancelled by user")
                                break

                        # Handle navigation
                        try:
                            async with page.expect_navigation(timeout=10000):
                                await submit_button.click()
                        except Exception as e:
                            print(f"Navigation error: {str(e)}")
                            if page_num == 1:
                                raise
                            print("Continuing to next page...")
                            break
                    else:
                        print("No valid submit button found - cannot proceed")
                        break

                    page_num += 1

                # Final verification
                await page.wait_for_timeout(2000)
                await page.screenshot(path='form_result.png', full_page=True)

            finally:
                await browser.close()


async def main(url):
    try:
        # Initialize RAG components
        embeddings = get_embedding_model(EMBEDDING_MODEL_NAME)
        vector_store = create_or_load_vector_store(
            "docs",
            embeddings,
            FAISS_INDEX_PATH
        )

        if not vector_store:
            raise RuntimeError("Failed to initialize vector store")

        rag_chain = create_rag_chain(vector_store, MISTRAL_MODEL)
        print("RAG system initialized successfully")

        # Initialize and execute form filling
        web_form = WebFormAgent(rag_chain)
        print("\nSTARTING FORM FILL PROCESS...")
        await web_form.fill_form_with_playwright(url)
        print("\nFORM FILL PROCESS COMPLETED")

    except Exception as e:
        print(f"\nERROR IN MAIN PROCESS: {str(e)}")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('url', help='Target URL')
    args = parser.parse_args()

    try:
        asyncio.run(main(args.url))
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        exit(1)
