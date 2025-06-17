# utils.py
from tenacity import retry, stop_after_attempt, wait_exponential
import logging


def setup_logging():
    logger = logging.getLogger("form_filler")
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler("form_filler.log")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def safe_model_load(model_name: str):
    return AutoModel.from_pretrained(model_name)

# pdf_utils.py


def sanitize_filename(filename: str) -> str:
    """Sanitize output filenames"""
    return "".join(c for c in filename if c.isalnum() or c in ('-', '_')).rstrip()
