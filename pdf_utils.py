import logging
import os
import fitz
from typing import List, Dict
from config import PDFConfig

logger = logging.getLogger(__name__)


class PDFFiller:
    def __init__(self, config: PDFConfig):
        self.config = config
        os.makedirs(self.config.output_dir, exist_ok=True)

    def fill_pdf(self, input_path: str, fields: List[Dict]) -> str:
        output_path = os.path.join(
            self.config.output_dir,
            os.path.basename(input_path)
        )

        try:
            doc = fitz.open(input_path)
            page = doc[0]

            for field in fields:
                if self._validate_field(field, page):
                    self._insert_text(page, field)

            doc.save(output_path, garbage=4, deflate=True)
            logger.info(f"Successfully saved: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"PDF filling failed: {str(e)}")
            return ""

    def _validate_field(self, field: Dict, page) -> bool:
        rect = field.get("field_value_box")
        if not isinstance(rect, fitz.Rect):
            return False

        page_rect = page.rect
        return all([
            rect.x0 >= page_rect.x0,
            rect.y0 >= page_rect.y0,
            rect.x1 <= page_rect.x1,
            rect.y1 <= page_rect.y1,
            rect.width > 0,
            rect.height > 0
        ])

    def _insert_text(self, page, field: Dict):
        rect = field["field_value_box"]
        text = field["answer"]

        page.insert_textbox(
            rect,
            text,
            fontname=self.config.font_name,
            fontsize=self.config.font_size,
            align=fitz.TEXT_ALIGN_LEFT
        )
