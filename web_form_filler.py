from playwright.async_api import Page, async_playwright
from typing import Dict, Any, Optional
import argparse
import asyncio
from ultralytics import YOLO


class PlaywrightFieldLocator:
    """Locates form fields using Playwright's built-in locators"""

    def __init__(self, page: Page):
        self.page = page
        self.field_heuristics = {
            'first_name': ['First Name', 'firstName', 'fname'],
            'last_name': ['Last Name', 'lastName', 'lname'],
            'email': ['Email', 'email', 'userEmail'],
            'phone': ['Phone', 'tel', 'phone'],
            'address': ['Address', 'street', 'addr'],
        }

    async def locate_fields(self) -> Dict[str, Any]:
        """Return dictionary of found field elements"""
        fields = {}

        for field_type, identifiers in self.field_heuristics.items():
            for identifier in identifiers:
                element = self.page.get_by_label(identifier, exact=True)
                if await element.count() == 0:
                    element = self.page.get_by_placeholder(
                        identifier, exact=True)
                if await element.count() == 0:
                    element = self.page.locator(f'[name="{identifier}"]')

                if await element.count() > 0:
                    fields[field_type] = element.first
                    break

        return fields


class CVFieldLocator:
    """Computer Vision fallback using YOLOv8 for field detection"""

    def __init__(self, page: Page, model_path: str):
        self.page = page
        self.model = self.load_model(model_path)

    def load_model(self, model_path: str):
        """Initialize YOLO model"""
        return YOLO(model_path)

    async def locate_fields(self, target_fields: list) -> Dict[str, Any]:
        """CV-based field detection using YOLO"""
        fields = {}

        # Take screenshot of current page
        screenshot_path = 'temp_screenshot.png'

        # Ensure all content loads
        await self.page.wait_for_load_state('networkidle')
        await self.page.wait_for_timeout(1000)  # Extra buffer

        # Scroll through entire page
        await self.page.evaluate("""async () => {
            await new Promise(resolve => {
                window.scrollTo(0, 0);
                let currentPos = 0;
                const scrollInterval = setInterval(() => {
                    window.scrollBy(0, window.innerHeight);
                    currentPos += window.innerHeight;
                    if (currentPos >= document.body.scrollHeight) {
                        clearInterval(scrollInterval);
                        resolve();
                    }
                }, 500);
            });
        }""")

        # Capture full page
        await self.page.screenshot(
            path=screenshot_path,
            full_page=True,
            animations="disabled",
            mask=[self.page.locator("header"), self.page.locator(
                ".cookie-banner")],  # Mask problematic elements
            timeout=60000
        )

        # Detect fields using YOLO model
        results = self.model.predict(screenshot_path)
        # print('results: ', results)

        for result in results:
            # print('results: ', result)
            for box in result.boxes:
                class_id = int(box.cls)
                field_type = result.names[class_id]
                if field_type in target_fields:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # Get device pixel ratio
                    dpr = await self.page.evaluate('window.devicePixelRatio')
                    center_x /= dpr
                    center_y /= dpr

                    # Get element at detected coordinates
                    element = await self.page.evaluate_handle(
                        f"document.elementFromPoint({center_x}, {center_y})"
                    )

                    # DOM traversal to find actual input element
                    real_input = await element.evaluate_handle('''el => {
                        return el.closest('label')?.control || 
                            el.querySelector('input, textarea, select') || 
                            el;
                    }''')

                    # Check if element is actually an input
                    tag_name = await real_input.evaluate('el => el.tagName.toLowerCase()')
                    is_input_like = tag_name in ['input', 'textarea', 'select']

                    if await real_input.is_visible() and is_input_like:
                        fields[field_type] = real_input



        return fields


class FormFiller:
    """Main class handling form filling with fallback strategies"""

    def __init__(self, page: Page, rag_retriever: Any, cv_model_path: Optional[str] = None):
        self.page = page
        self.rag = rag_retriever
        self.playwright_locator = PlaywrightFieldLocator(page)
        self.cv_locator = CVFieldLocator(
            page, cv_model_path) if cv_model_path else None

    async def fill_form(self) -> None:
        """Main form filling workflow"""
        form_data = self.rag.get_form_data()
        fields = await self.playwright_locator.locate_fields()

        missing_fields = [ft for ft in form_data.keys() if ft not in fields]

        if missing_fields and self.cv_locator:
            cv_fields = await self.cv_locator.locate_fields(missing_fields)
            print('cv_fields: ', cv_fields)
            fields.update(cv_fields)

        for field_type, element in fields.items():
            if field_type in form_data:
                await element.fill(form_data[field_type])

        remaining_missing = [ft for ft in form_data.keys() if ft not in fields]
        if remaining_missing:
            print(
                f"Warning: Could not locate fields: {', '.join(remaining_missing)}")


async def main(url: str, model_path: Optional[str] = None, headless: bool = True):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()

        try:
            await page.goto(url)
            await page.wait_for_load_state('networkidle')

            class DummyRAG:
                def get_form_data(self):
                    return {
                        'first_name': 'John',
                        'last_name': 'Doe',
                        'email': 'john@example.com',
                        'phone': '555-1234'
                    }

            filler = FormFiller(
                page=page,
                rag_retriever=DummyRAG(),
                cv_model_path=model_path
            )

            await filler.fill_form()
            await page.wait_for_timeout(2000)  # Visual verification
        finally:
            await browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Automated Form Filler')
    parser.add_argument('--url', required=True, help='URL of the form page')
    parser.add_argument('--model', help='Path to YOLO model')
    parser.add_argument('--visible', action='store_false',
                        dest='headless', help='Run browser in visible mode')

    args = parser.parse_args()

    asyncio.run(main(
        url=args.url,
        model_path=args.model,
        headless=args.headless
    ))
