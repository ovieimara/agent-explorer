import asyncio
import datetime
from typing import List, Dict
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from playwright.async_api import Page, ElementHandle


class RagAgent:
    def __init__(self, retriever, bert_extractor, llm, prompt):
        self.chain = self._create_rag_chain(
            retriever, bert_extractor, llm, prompt)

    def _create_rag_chain(self, retriever, bert_extractor, llm, prompt):
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnableLambda(self._format_context)
            | {
                "bert_answer": RunnableLambda(bert_extractor),
                "question": itemgetter("question"),
                "context": itemgetter("context")
            }
            | {
                "answer": RunnableLambda(self._generate_answer),
                "source": RunnableLambda(self._detect_source)
            }
        )

    def _format_context(self, x):
        return {
            "context": " ".join([doc.page_content for doc in x["context"]]),
            "question": x["question"]
        }

    def _generate_answer(self, x):
        return x["bert_answer"] or self.llm.invoke(
            self.prompt.format(
                question=x["question"],
                context=x["context"]
            )
        ).content.strip()

    def _detect_source(self, x):
        return "BERT" if x["bert_answer"] else "LLM"

    async def get_field_value(self, question: str) -> Dict:
        response = await self.chain.ainvoke(question)
        return {
            "value": response["answer"],
            "source": response["source"],
            "timestamp": datetime.datetime.now().isoformat()
        }


class FormLabelingAgent:
    def __init__(self, field_type_map: dict):
        self.field_type_map = field_type_map
        self.field_selectors = [
            'input:not([type="submit"]):visible',
            'textarea:visible',
            'select:visible',
            '[role="textbox"]:visible',
            '[contenteditable="true"]:visible'
        ]

    @staticmethod
    def default_field_map():
        return {
            'email': {'label': 'Email', 'question': 'Email address for contact'},
            'password': {'label': 'Password', 'question': 'Secure password for account'},
            # ... other defaults
        }

    def _generate_label(self, metadata: Dict) -> str:
        field_type = metadata['type']
        return self.field_type_map.get(
            field_type,
            self.field_type_map.get('*', {})
        ).get('label', 'Form Field')

    async def locate_fields(self, page: Page) -> List[Dict]:
        """Ensure proper Page object handling"""
        fields = []
        for selector in self.field_selectors:
            elements = await page.query_selector_all(selector)
            for element in elements:
                if await element.is_visible():
                    field_data = await self._get_field_metadata(element)
                    field_data["element_handle"] = element  # Preserve handle
                    fields.append(field_data)
        return fields

    async def _get_field_metadata(self, element: ElementHandle) -> Dict:
        """Maintain proper ElementHandle typing"""
        return {
            "element": element,  # Preserve original element reference
            "tag": await element.evaluate('el => el.tagName.toLowerCase()'),
            "type": await element.get_attribute('type') or 'text',
            "name": await element.get_attribute('name'),
            "id": await element.get_attribute('id'),
            "placeholder": await element.get_attribute('placeholder'),
            "aria_label": await element.get_attribute('aria-label'),
            "xpath": await self._get_xpath(element),
            "label": self._generate_label(await element.get_attribute('type'))
        }

    def _generate_label(self, field_type: str) -> str:
        """Use FIELD_TYPE_MAP for label generation"""
        return self.field_type_map.get(
            field_type,
            self.field_type_map.get('*', {'label': 'Form Field'})
        )['label']

    async def _get_xpath(self, element: ElementHandle) -> str:
        """Properly formatted XPath generator"""
        return await element.evaluate('''(element) => {
            function getPathTo(e) {
                if (e.id !== '') {
                    return '//' + e.tagName.toLowerCase() + '[@id="' + e.id + '"]';
                }
                if (e === document.body) {
                    return e.tagName.toLowerCase();
                }

                let ix = 0;
                const siblings = e.parentNode.childNodes;

                for (let i = 0; i < siblings.length; i++) {
                    const sibling = siblings[i];
                    if (sibling === e) {
                        return getPathTo(e.parentNode) + '/' + e.tagName.toLowerCase() + 
                            '[' + (ix + 1) + ']';
                    }
                    if (sibling.nodeType === 1 && sibling.tagName === e.tagName) {
                        ix++;
                    }
                }
                return ''; // Default return
            }
            
            try {
                return getPathTo(element);
            } catch (error) {
                return ''; // Fail gracefully
            }
        }''')

    def _filter_fields(self, fields: List[Dict]) -> List[Dict]:
        return [f for f in fields if f['type'] not in ['hidden', 'submit']]


class FormLabelingAgent:
    async def locate_fields(self, page: Page) -> List[Dict]:
        """Now properly async"""
        fields = []
        for selector in self.field_selectors:
            elements = await page.query_selector_all(selector)
            for element in elements:
                if await element.is_visible():
                    fields.append(await self._get_field_metadata(element))
        return fields


class FormFillingAgent:
    async def fill_form(self, page: Page, answers: List[Dict]):
        """Async form filling"""
        for answer in answers:
            element = await page.query_selector(answer["selector"])
            if element:
                await element.fill(answer["value"])
                await page.wait_for_timeout(500)  # Brief delay between fills

    async def handle_submission(self, page: Page) -> bool:
        """Async submission handler"""
        submit_button = await page.query_selector('input[type="submit"], button[type="submit"]')
        if submit_button:
            await submit_button.click()
            try:
                await page.wait_for_navigation(timeout=5000)
                return True
            except:
                return False
        return False
    

async def main(url):
    # Initialize components
    rag_agent = RagAgent(...)
    label_agent = FormLabelingAgent(FIELD_TYPE_MAP)
    fill_agent = FormFillingAgent()

    # Create orchestrator
    orchestrator = FormOrchestrator(rag_agent, label_agent, fill_agent)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)

        # Execute workflow
        result = await orchestrator.process_form(page)
        print(f"Final state: {result}")
        await browser.close()
