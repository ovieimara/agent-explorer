# form_filler.py (LayoutLMv3 Analysis Version with Debugging - OCR Fix)

import os
import fitz  # PyMuPDF
import pytesseract  # For OCR
from PIL import Image  # For handling images
from form_analyzer import PDFFormAnalyzer
from transformers import AutoProcessor, AutoModelForTokenClassification,  DonutProcessor, VisionEncoderDecoderModel  # LayoutLMv3
import torch  # PyTorch
from operator import itemgetter
import traceback  # For detailed error printing

# Import functions from our RAG script
from rag import get_embedding_model, create_or_load_vector_store, create_rag_chain

# --- Configuration ---
# RAG Configuration
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLAMA3_MODEL = "llama3:8b"
DEEPSEEK_MODEL = "deepseek-r1:8b"  # Adjust if you pulled a different tag
MISTRAL_MODEL = "mistral:7b-instruct"  # Adjust if you pulled a different tag
GEMMA_MODEL = "gemma3:12b-it-qat"  # Adjust if you pulled a different tag
QWEN_MODEL = "qwen2:7b"

# File Paths
# !!! CHANGE THIS to the PDF form you want to fill !!!
TARGET_PDF_FORM = "target_form.pdf"
OUTPUT_FILLED_PDF = "output/filled_form_layoutlm_debug.pdf"  # Keep debug name
OUTPUT_DIR = "output"

# LayoutLMv3 Configuration
LAYOUTLM_MODEL_NAME = "nielsr/layoutlmv3-finetuned-funsd"
ANSWER_BOX_WIDTH_FRACTION = 0.4
VERTICAL_ALIGNMENT_THRESHOLD = 10
# Tesseract OCR confidence threshold (0-100, -1 to disable)
OCR_CONFIDENCE_THRESHOLD = 50  # Ignore words with lower confidence

# --- Helper Functions --- (normalize_box, unnormalize_box - unchanged)


# def analyze_pdf_form_layoutlm(file_path, processor, model):
#     """
#     Analyzes a PDF form's first page using LayoutLMv3.
#     Performs OCR separately and passes results to the processor.
#     Includes enhanced debugging prints.
#     """
#     print(
#         f"\n--- Analyzing PDF Form with LayoutLMv3 ({LAYOUTLM_MODEL_NAME}): {file_path} ---")
#     if not os.path.exists(file_path):
#         print(f"DEBUG: Error - Target PDF form not found at: {file_path}")
#         return None, 0, 0

#     extracted_fields = []
#     page_width_pts, page_height_pts = 0, 0

#     try:
#         # 1. Render PDF page to Image
#         print("DEBUG: Opening PDF for rendering...")
#         doc = fitz.open(file_path)
#         if doc.page_count == 0:
#             print("DEBUG: Error - PDF has no pages.")
#             doc.close()
#             return None, 0, 0

#         page = doc[0]
#         page_width_pts = page.rect.width
#         page_height_pts = page.rect.height
#         print(
#             f"DEBUG: PDF Page Dimensions (points): Width={page_width_pts}, Height={page_height_pts}")
#         pix = page.get_pixmap(dpi=300)
#         img = Image.frombytes(
#             "RGB", [pix.width, pix.height], pix.samples).convert("RGB")
#         page_width_px, page_height_px = img.size
#         print(
#             f"DEBUG: Rendered Image Dimensions (pixels): Width={page_width_px}, Height={page_height_px}")
#         doc.close()
#         print("DEBUG: PDF closed after rendering.")

#         # --- OCR Step ---
#         print("DEBUG: Performing OCR using pytesseract...")
#         try:
#             ocr_data = pytesseract.image_to_data(
#                 img, output_type=pytesseract.Output.DICT, lang='eng')
#             print(
#                 f"DEBUG: OCR complete. Found {len(ocr_data['text'])} potential text elements.")
#         except pytesseract.TesseractNotFoundError:
#             print("DEBUG: Error - Tesseract is not installed or not in your PATH.")
#             print(
#                 "Please install Tesseract OCR engine (e.g., 'brew install tesseract' on macOS).")
#             return None, page_width_pts, page_height_pts
#         except Exception as ocr_error:
#             print(f"DEBUG: Error during pytesseract OCR: {ocr_error}")
#             traceback.print_exc()
#             return None, page_width_pts, page_height_pts

#         words = []
#         boxes = []
#         print("DEBUG: Extracting words and boxes from OCR data...")
#         n_boxes = len(ocr_data['level'])
#         for i in range(n_boxes):
#             # Check confidence level
#             conf = int(ocr_data['conf'][i])
#             # -1 means not applicable (e.g., page breaks)
#             if conf < OCR_CONFIDENCE_THRESHOLD and conf != -1:
#                 # print(f"DEBUG: Skipping low confidence word: '{ocr_data['text'][i]}' (conf: {conf})")
#                 continue

#             # Check if it's a word-level detection
#             if ocr_data['level'][i] == 5:  # Level 5 typically corresponds to word
#                 text = ocr_data['text'][i].strip()
#                 if text:  # Only add non-empty words
#                     (x, y, w, h) = (ocr_data['left'][i], ocr_data['top']
#                                     [i], ocr_data['width'][i], ocr_data['height'][i])
#                     # Tesseract box: [left, top, width, height]
#                     # Processor expects: [x_min, y_min, x_max, y_max] normalized 0-1000
#                     box = [x, y, x + w, y + h]
#                     normalized_bbox = normalize_box(
#                         box, page_width_px, page_height_px)

#                     # Final check for valid normalized coordinates
#                     if all(0 <= coord <= 1000 for coord in normalized_bbox):
#                         boxes.append(normalized_bbox)
#                         words.append(text)
#                     else:
#                         print(
#                             f"DEBUG: Skipping box with invalid normalized coordinates: {normalized_bbox}")

#                     words.append(text)
#                     boxes.append(normalized_bbox)
#                     # print(f"DEBUG: OCR Word: '{text}', Norm Box: {normalized_bbox}") # Verbose

#         if not words:
#             print(
#                 "DEBUG: Error - No words found after OCR processing. Cannot proceed with LayoutLM.")
#             return None, page_width_pts, page_height_pts
#         print(
#             f"DEBUG: Extracted {len(words)} words with confidence >= {OCR_CONFIDENCE_THRESHOLD}.")

#         # --- Prepare input for LayoutLMv3 using OCR results ---
#         print("DEBUG: Preparing LayoutLMv3 input using extracted words and boxes...")
#         # Pass the image, words, and boxes explicitly
#         # Remove the apply_ocr=True argument
#         encoding = processor(img, text=words, boxes=boxes, return_tensors="pt",
#                              truncation=True, padding="max_length")  # Add truncation/padding

#         device = torch.device(
#             "mps" if torch.backends.mps.is_available() else "cpu")
#         print(f"DEBUG: Using device: {device}")
#         model.to(device)
#         for key, tensor in encoding.items():
#             encoding[key] = tensor.to(device)
#         print("DEBUG: Input tensors prepared for model.")

#         # --- Run Inference ---
#         print("DEBUG: Running LayoutLMv3 model inference...")
#         with torch.no_grad():
#             outputs = model(**encoding)
#         logits = outputs.logits
#         predictions = logits.argmax(-1).squeeze().tolist()
#         print("DEBUG: Model inference complete.")

#         # --- Post-process ---
#         print("DEBUG: Post-processing model output...")
#         # Get token boxes directly from the encoding THIS TIME (as they correspond to the input words)
#         token_boxes_normalized = encoding["bbox"].squeeze().tolist()
#         # Still needed to map tokens to input words
#         word_ids = encoding.word_ids(batch_index=0)

#         try:
#             id2label = model.config.id2label
#             labels = [id2label[pred] for pred in predictions]
#         except KeyError as e:
#             print(
#                 f"DEBUG: Error - Prediction index {e} not found in model's id2label config.")
#             print(f"DEBUG: Model config id2label: {model.config.id2label}")
#             labels = ['O'] * len(predictions)
#         except AttributeError:
#             print("DEBUG: Error - Model config does not have id2label attribute.")
#             return None, page_width_pts, page_height_pts

#         # Group tokens by word ID (which now corresponds to the index in our input `words` list)
#         word_level_data = {}
#         print("DEBUG: Grouping tokens by word ID...")
#         current_word_idx = -1
#         for token_idx, word_id in enumerate(word_ids):
#             if word_id is None:
#                 continue  # Skip special tokens

#             # Check if word_id is within the bounds of our OCR'd words list
#             if word_id != current_word_idx:
#                 current_word_idx = word_id
#                 if current_word_idx >= len(words):
#                     # This might happen with truncation/padding, skip these tokens
#                     # print(f"DEBUG: Skipping token with word_id {current_word_idx} beyond OCR word count {len(words)}")
#                     continue

#                 if word_id not in word_level_data:
#                     word_level_data[word_id] = {
#                         'text': words[word_id], 'boxes_norm': [], 'labels': set()}

#                 # Store the normalized box for this token
#                 word_level_data[word_id]['boxes_norm'].append(
#                     token_boxes_normalized[token_idx])
#                 word_level_data[word_id]['labels'].add(labels[token_idx])

#         final_words = []
#         print("DEBUG: Consolidating word data...")
#         for word_id in sorted(word_level_data.keys()):
#             word_data = word_level_data[word_id]
#             if not word_data['boxes_norm']:
#                 # Skip if no boxes were collected (e.g., due to skipping)
#                 continue

#             # Consolidate normalized boxes
#             min_x0_n = min(b[0] for b in word_data['boxes_norm'])
#             min_y0_n = min(b[1] for b in word_data['boxes_norm'])
#             max_x1_n = max(b[2] for b in word_data['boxes_norm'])
#             max_y1_n = max(b[3] for b in word_data['boxes_norm'])
#             consolidated_box_norm = [min_x0_n, min_y0_n, max_x1_n, max_y1_n]

#             # Determine primary label
#             primary_label = 'O'
#             labels_present = word_data['labels']
#             if any(l in labels_present for l in ['B-QUESTION', 'I-QUESTION']):
#                 primary_label = 'QUESTION'
#             elif any(l in labels_present for l in ['B-ANSWER', 'I-ANSWER']):
#                 primary_label = 'ANSWER'
#             elif any(l in labels_present for l in ['B-HEADER', 'I-HEADER']):
#                 primary_label = 'HEADER'

#             # Unnormalize box from 0-1000 back to image pixel coordinates
#             pixel_box = unnormalize_box(
#                 consolidated_box_norm, page_width_px, page_height_px)
#             # Convert pixel coordinates to PDF points
#             pdf_box = [
#                 pixel_box[0] * (page_width_pts / page_width_px), pixel_box[1] *
#                 (page_height_pts / page_height_px),
#                 pixel_box[2] * (page_width_pts / page_width_px), pixel_box[3] *
#                 (page_height_pts / page_height_px),
#             ]

#             final_words.append({
#                 "text": word_data['text'], "label": primary_label, "pdf_box": pdf_box,
#                 "center_y": (pdf_box[1] + pdf_box[3]) / 2
#             })

#         # --- Heuristic: Find 'Question' labels and infer 'Answer' boxes ---
#         print("DEBUG: Inferring answer box locations...")
#         found_fields = {}
#         question_words = [
#             word for word in final_words if word['label'] == 'QUESTION']
#         question_words.sort(key=itemgetter("center_y"))
#         print(f"DEBUG: Found {len(question_words)} words labeled as QUESTION.")

#         for qw_idx, qw in enumerate(question_words):
#             q_text = qw['text'].strip()
#             qx0, qy0, qx1, qy1 = qw['pdf_box']
#             clean_q_text = q_text.replace(":", "").strip()
#             if not clean_q_text:
#                 continue

#             print(
#                 f"DEBUG: Processing QUESTION word #{qw_idx}: '{clean_q_text}' at {qw['pdf_box']}")

#             ans_x0 = qx1 + 5
#             ans_y0 = qy0
#             ans_x1 = min(ans_x0 + (page_width_pts *
#                          ANSWER_BOX_WIDTH_FRACTION), page_width_pts - 5)
#             ans_y1 = qy1
#             answer_box_rect = fitz.Rect(ans_x0, ans_y0, ans_x1, ans_y1)

#             if answer_box_rect.is_valid and answer_box_rect.width > 0 and answer_box_rect.height > 0:
#                 if clean_q_text.lower() not in found_fields:
#                     found_fields[clean_q_text.lower()] = {
#                         "field_name": clean_q_text,
#                         "field_value_box": answer_box_rect
#                     }
#                     print(
#                         f"DEBUG:   -> Inferred Answer Box: {answer_box_rect}")
#                 else:
#                     print(
#                         f"DEBUG:   -> Skipping duplicate field name: '{clean_q_text.lower()}'")
#             else:
#                 print(
#                     f"DEBUG:   -> Warning: Invalid inferred box: {answer_box_rect}")

#         extracted_fields = list(found_fields.values())
#         print(
#             f"DEBUG: Finished analysis. Found {len(extracted_fields)} potential fields.")

#     except ImportError as e:
#         print(
#             f"DEBUG: Import Error during analysis: {e}. Check requirements.txt.")
#         traceback.print_exc()
#         return None, page_width_pts, page_height_pts
#     except fitz.FileNotFoundError:
#         print(f"DEBUG: Error - PyMuPDF could not find the file: {file_path}")
#         return None, 0, 0
#     except Exception as e:
#         print(f"DEBUG: Error during LayoutLMv3 analysis: {e}")
#         traceback.print_exc()
#         return None, page_width_pts, page_height_pts

#     if not extracted_fields:
#         print("DEBUG: Warning - No form fields were ultimately identified after analysis.")

#     return extracted_fields, page_width_pts, page_height_pts


# --- PDF Filling Function (Unchanged) ---
def fill_pdf_form(input_pdf_path, output_pdf_path, fields_to_fill):
    """ Fills PDF form using PyMuPDF. Includes enhanced debugging. """
    print(f"\n--- Filling PDF: {output_pdf_path} ---")
    print(f"DEBUG: Attempting to fill {len(fields_to_fill)} fields.")
    if not fields_to_fill:
        print("DEBUG: No fields provided to fill. Skipping PDF writing.")
        return False

    doc = None
    try:
        print(f"DEBUG: Opening source PDF for writing: {input_pdf_path}")
        doc = fitz.open(input_pdf_path)
        if not doc or doc.page_count == 0:
            print(
                f"DEBUG: Error - Could not open input PDF or it has no pages: {input_pdf_path}")
            if doc:
                doc.close()
            return False

        page = doc[0]
        stamped_count = 0

        for field_idx, field in enumerate(fields_to_fill):
            field_name = field.get("field_name", f"Field_{field_idx}")
            value_box = field.get("field_value_box")
            answer = field.get("answer", "")
            print(
                f"DEBUG: Processing field {field_idx+1}/{len(fields_to_fill)}: '{field_name}'")
            print(f"DEBUG:   Answer from RAG: '{answer}'")
            print(f"DEBUG:   Target Box: {value_box}")

            if value_box and isinstance(value_box, fitz.Rect) and value_box.is_valid and answer and answer != "Information not found in context" and "Error during RAG query" not in answer:
                # Calculate dynamic padding based on field width (2% of width, min 5pt, max 20pt)
                field_width = value_box.x1 - value_box.x0
                left_padding = max(5, max(50, int(field_width * 0.01)))
                # fontsize = 5
                # text_v_adjust = fontsize * 0.3
                # insert_point = fitz.Point(
                #     value_box.x0 + 2, value_box.y0 + text_v_adjust)
                # Choose a compact font
                fontname = "tiro"  # Experiment with this
                font = fitz.Font(fontname)

                # Calculate max usable fontsize for this box
                box_height = value_box.height
                max_fsize = box_height / (font.ascender - font.descender)
                fontsize = min(max_fsize * 0.95, 12)  # Cap at 8pt for sanity

                # Calculate baseline position
                baseline_y = value_box.y0 + (font.ascender * fontsize)
                start_x = value_box.x0 + left_padding  # Left padding

                # Draw debug box (red) and baseline (blue)
                # page.draw_rect(value_box, color=(1, 0, 0), width=0.5)
                # page.draw_line(
                #     fitz.Point(value_box.x0, baseline_y),
                #     fitz.Point(value_box.x1, baseline_y),
                #     color=(0, 0, 1),
                #     width=0.5
                # )

                print(
                    f"DEBUG:   Attempting to insert text '{answer[:30]}...' into box {value_box}")
                # Draw debug visualization
                page.draw_rect(value_box, color=(1, 0, 0),
                               width=0.5)  # Field boundary
                page.draw_line(
                    fitz.Point(start_x, value_box.y0),
                    fitz.Point(start_x, value_box.y1),
                    color=(0, 0.5, 0), width=0.5
                )  # Padding visualization

                try:
                    text_to_insert = str(answer)

                    # rc = page.insert_textbox(
                    #     value_box, text_to_insert, fontsize=fontsize, fontname="helv", color=(0, 0, 0),
                    #     align=fitz.TEXT_ALIGN_LEFT
                    # )
                    # Insert text with precise positioning
                    rc = page.insert_text(
                        point=(start_x, baseline_y),
                        text=text_to_insert,
                        fontsize=fontsize,
                        fontname=fontname,
                        color=(0, 0, 0)
                    )
                    if rc < 0:
                        print(
                            f"DEBUG:   -> Warning: Text overflow reported by PyMuPDF (rc={rc}).")
                    else:
                        print(
                            f"DEBUG:   -> Text insertion successful (rc={rc}).")
                    stamped_count += 1
                except Exception as insert_error:
                    print(
                        f"DEBUG:   -> Error during page.insert_textbox for field '{field_name}': {insert_error}")
                    traceback.print_exc()
            else:
                print(
                    f"DEBUG:   -> Skipping insertion for field '{field_name}' (invalid box, empty answer, or RAG error).")

        if stamped_count > 0:
            print(
                f"\nDEBUG: Attempting to save filled PDF to {output_pdf_path} ({stamped_count} fields stamped)...")
            os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
            doc.save(output_pdf_path, garbage=4, deflate=True)
            print(f"DEBUG: Filled PDF saved successfully.")
        else:
            print("\nDEBUG: No fields were actually stamped with answers. Output PDF may be identical to input or empty if save skipped.")

        doc.close()
        print("DEBUG: Source PDF closed after processing.")
        return True

    except Exception as e:
        print(f"DEBUG: Error during PDF filling process: {e}")
        traceback.print_exc()
        if doc:
            try:
                doc.close()
                print("DEBUG: Source PDF closed after error.")
            except:
                pass
        return False


# --- Main Execution --- (Initialize RAG, Analyze, Query RAG, Fill PDF)
if __name__ == "__main__":
    print("--- Form Filling Agent (LayoutLMv3 Analysis - Debug Mode) ---")

    # --- Phase 0: Load LayoutLMv3 Model ---
    print(
        f"\nDEBUG: Loading LayoutLMv3 model and processor: {LAYOUTLM_MODEL_NAME}...")
    layoutlm_processor = None
    layoutlm_model = None
    try:
        layoutlm_processor = AutoProcessor.from_pretrained(
            LAYOUTLM_MODEL_NAME, apply_ocr=False)  # Removed apply_ocr here
        layoutlm_model = AutoModelForTokenClassification.from_pretrained(
            LAYOUTLM_MODEL_NAME)
        print("DEBUG: LayoutLMv3 model loaded successfully.")
    except Exception as e:
        print(f"DEBUG: Error loading LayoutLMv3 model/processor: {e}")
        print("DEBUG: Ensure model name is correct and you have internet access for first download.")
        traceback.print_exc()
        exit()

    # --- Phase 1: Initialize RAG System ---
    print("\nDEBUG: Initializing RAG components...")
    rag_chain = None
    try:
        embeddings = get_embedding_model(EMBEDDING_MODEL_NAME)
        vector_store = create_or_load_vector_store(
            "docs", embeddings, FAISS_INDEX_PATH)
        if vector_store:
            rag_chain = create_rag_chain(vector_store, QWEN_MODEL)
        else:
            print("DEBUG: Error - Failed to load or create vector store.")
            exit()
        if not rag_chain:
            print("DEBUG: Error - Failed to create RAG chain.")
            exit()
        print("DEBUG: RAG system ready.")
    except Exception as e:
        print(f"DEBUG: Error initializing RAG system: {e}")
        traceback.print_exc()
        exit()

    # --- Phase 2: Analyze the Target PDF Form (LayoutLMv3) ---
    # Pass the loaded processor and model
    # form_fields, page_w, page_h = analyze_pdf_form_layoutlm(
    #     TARGET_PDF_FORM, layoutlm_processor, layoutlm_model)

        # Initialize Donut optionally
    donut_processor, donut_model = None, None
    try:
        donut_processor = DonutProcessor.from_pretrained(
            "naver-clova-ix/donut-base",  use_fast=True)
        donut_model = VisionEncoderDecoderModel.from_pretrained(
            "naver-clova-ix/donut-base")
    except Exception as e:
        print(f"⚠️ Couldn't load Donut: {e}")

    # Create analyzer
    analyzer = PDFFormAnalyzer(
        layoutlm_processor=layoutlm_processor,
        layoutlm_model=layoutlm_model,
        donut_processor=donut_processor,
        donut_model=donut_model
    )

    # analyzer = PDFFormAnalyzer(layoutlm_processor, layoutlm_model)
    form_fields, page_w, page_h = analyzer.analyze(TARGET_PDF_FORM)
    # print("form_fields:", form_fields)

    if form_fields is None:
        print("DEBUG: PDF analysis failed. Exiting.")
        exit()
    elif not form_fields:
        print("DEBUG: PDF analysis completed, but no fields were identified. Exiting as there's nothing to fill.")
        exit()

    # --- Phase 3: Query RAG and Prepare Data for Filling ---
    print("\n--- Querying RAG for Form Fields ---")
    fields_to_fill_list = []
    print(
        f"DEBUG: Starting RAG queries for {len(form_fields)} identified fields...")
    for field_idx, field in enumerate(form_fields):
        field_label = field.get("field_name")
        field_box = field.get("field_value_box")

        print(f"DEBUG: Querying for Field {field_idx+1}: '{field_label}'")
        if field_label and field_box:
            question = f"What is my {field_label}?"
            print(f"DEBUG:   RAG Question: '{question}'")
            try:
                answer_result = rag_chain.invoke(question)
                print(f"DEBUG:   RAG Answer Raw: '{answer_result}'")

                # Extract the answer value from the result dictionary
                answer_value = answer_result.get("answer", "")

                if isinstance(answer_value, str):
                    clean_answer = answer_value.strip()
                else:
                    # Handle AIMessage (LLM fallback) by extracting content
                    clean_answer = getattr(answer_value, "content", "").strip()

                fields_to_fill_list.append({
                    "field_name": field_label,
                    "field_value_box": field_box,
                    "answer": clean_answer
                })
            except Exception as e:
                print(
                    f"DEBUG:   -> Error invoking RAG chain for '{field_label}': {e}")
                traceback.print_exc()
                fields_to_fill_list.append({
                    "field_name": field_label,
                    "field_value_box": field_box,
                    "answer": "Error during RAG query"
                })
        else:
            print(
                f"DEBUG:   Skipping RAG query - missing label or inferred box: {field}")

    # --- Phase 3: Fill the PDF ---
    if fields_to_fill_list:
        # fields_to_fill_list = [{'field_name': 'Name', 'field_value_box': fitz.Rect(
        #     110.35925, 74.938, 348.45925, 83.358), 'answer': 'Oviemuno Imara'}]
        # print("fields_to_fill_list: ", fields_to_fill_list)
        print(
            f"DEBUG: Proceeding to fill PDF with {len(fields_to_fill_list)} field answers.")
        fill_pdf_form(
            TARGET_PDF_FORM,
            OUTPUT_FILLED_PDF,
            fields_to_fill_list
        )
    else:
        print("\nDEBUG: No valid RAG answers obtained to fill. Skipping PDF writing.")

    print("\n--- Form Filling Process Complete ---")
