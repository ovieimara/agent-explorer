import logging
from config import Config
from document_ai import DocumentAnalyzer
from rag_system import RAGSystem
from pdf_utils import PDFFiller


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("form_filler.log"),
            logging.StreamHandler()
        ]
    )


def main():
    config = Config()
    setup_logging()

    try:
        # Initialize components
        analyzer = DocumentAnalyzer(config.document_ai)
        rag = RAGSystem(config.rag)
        filler = PDFFiller(config.pdf)

        # Analyze PDF
        fields = analyzer.analyze_pdf("target_form.pdf")
        if fields:
            print(f"Detected {len(fields)} form fields:")
            for field in fields:
                print(f"- {field['field_name']} ({field['confidence']:.1%})")

        else:
            print("No fields detected - check OCR quality or model configuration")
        # if not fields:
        #     logging.error("No fields detected")
        #     return

        # Query RAG
        answers = []
        for field in fields:
            answer = rag.query(field["question"])
            if answer:
                answers.append({
                    **field,
                    "answer": answer
                })

        # Fill PDF
        if answers:
            output_path = filler.fill_pdf("input_form.pdf", answers)
            if output_path:
                logging.info(f"Successfully created: {output_path}")
            else:
                logging.error("Failed to generate output PDF")
        else:
            logging.warning("No valid answers to fill")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
