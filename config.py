# config.py
from dataclasses import dataclass, field


@dataclass
class DocumentAIConfig:
    # Add new parameters
    min_ocr_confidence: int = 60  # Reject OCR results below this %
    max_fields: int = 50          # Maximum fields to return per page
    ocr_timeout: int = 30         # Seconds per page for OCR


@dataclass
class RAGConfig:
    """Configuration for Retrieval-Augmented Generation system"""
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "models/llama-2-7b-chat.Q4_K_M.gguf"
    faiss_index_path: str = "faiss_index"
    temperature: float = 0.7
    max_tokens: int = 500
    faiss_parallel_mode: int = 3  # Optimized for Apple Silicon
    faiss_search_threads: int = 8  # Match M1 Pro core count
    context_window: int = 4096


@dataclass
class PDFConfig:
    """Configuration for PDF processing"""
    font_name: str = "helv"
    font_size: int = 12
    dpi: int = 300
    output_dir: str = "output"
    default_color: tuple = (0, 0, 0)  # RGB black
    line_spacing: float = 1.2


@dataclass
class Config:
    """Main configuration container"""
    document_ai: DocumentAIConfig = field(default_factory=DocumentAIConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    pdf: PDFConfig = field(default_factory=PDFConfig)

    @classmethod
    def create(cls, **kwargs):
        """Factory method for creating config with overrides"""
        return cls(
            document_ai=DocumentAIConfig(**kwargs.get("document_ai", {})),
            rag=RAGConfig(**kwargs.get("rag", {})),
            pdf=PDFConfig(**kwargs.get("pdf", {}))
        )

    def validate(self):
        """Validate configuration values"""
        assert 0 < self.rag.temperature <= 2.0, "Temperature must be between 0 and 2"
        assert self.pdf.font_size >= 6, "Font size too small"
        assert self.document_ai.dpi >= 150, "DPI too low for OCR accuracy"
