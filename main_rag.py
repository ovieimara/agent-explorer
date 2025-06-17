# main_rag.py (Updated with Tool Function)

import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

# --- NEW: Import BERT/QA components ---
# Ensure transformers is installed: pip install transformers[torch]
try:
    from transformers import pipeline, BertForQuestionAnswering, BertTokenizer
except ImportError:
    print("Warning: 'transformers' library not found or BERT components missing.")
    print("Please install it: pip install transformers[torch]")
    pipeline = None
    BertForQuestionAnswering = None
    BertTokenizer = None

# --- Configuration ---
DOCS_PATH = "docs"
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLAMA3_MODEL = "llama3:8b"  # LLM for fallback/generation
# BERT model for extractive QA (SQuAD fine-tuned)
BERT_QA_MODEL_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"
# Confidence threshold for BERT answers
BERT_SCORE_THRESHOLD = 0.3

# --- Global Variables for Lazy Loading ---
# Avoid loading models repeatedly if called multiple times
_vector_store = None
_rag_chain = None
_embeddings = None

# --- Reusable Functions ---


def get_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    """Initializes and returns the embedding model."""
    global _embeddings
    if _embeddings is None:
        print(f"Initializing embedding model: {model_name}")
        _embeddings = HuggingFaceEmbeddings(model_name=model_name)
        print("Embedding model initialized.")
    return _embeddings


def load_or_create_vector_store(docs_path, index_path, embedding_model):
    """Loads an existing FAISS index or creates a new one from documents."""
    global _vector_store
    if _vector_store is not None:
        print("DEBUG: Using already loaded vector store.")
        return _vector_store

    if os.path.exists(index_path):
        print(f"Loading existing FAISS index from: {index_path}")
        _vector_store = FAISS.load_local(
            index_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        print("FAISS index loaded.")
    else:
        # ... (Keep the index creation logic from previous main_rag.py) ...
        print("No existing index found. Creating new index...")
        print(f"Loading documents from: {docs_path}")
        loader = DirectoryLoader(
            docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader,
            recursive=True, show_progress=True
        )
        documents = loader.load()
        if not documents:
            print(
                f"Error: No documents found in '{docs_path}'. Cannot create index.")
            return None
        print(f"Loaded {len(documents)} document pages.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks.")
        if not chunks:
            print("Error: Failed to split documents into chunks.")
            return None
        print("Creating FAISS index from document chunks...")
        _vector_store = FAISS.from_documents(chunks, embedding_model)
        print("FAISS index created.")
        print(f"Saving FAISS index to: {index_path}")
        _vector_store.save_local(index_path)
        print("FAISS index saved.")

    return _vector_store


def create_rag_chain(vector_store, llm_model_name=LLAMA3_MODEL, bert_model_name=BERT_QA_MODEL_NAME):
    """Creates hybrid RAG chain with BERT for extraction and LLM for fallback"""
    global _rag_chain
    if _rag_chain is not None:
        print("DEBUG: Using already created RAG chain.")
        return _rag_chain

    if vector_store is None:
        print("Vector store is not available. Cannot create RAG chain.")
        return None

    # --- Initialize BERT QA model ---
    bert_qa_pipeline = None
    if pipeline and BertForQuestionAnswering and BertTokenizer:  # Check imports succeeded
        try:
            print(f"Loading BERT QA pipeline: {bert_model_name}")
            # Load model and tokenizer explicitly first for better error handling
            bert_model = BertForQuestionAnswering.from_pretrained(
                bert_model_name)
            bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            bert_qa_pipeline = pipeline(
                "question-answering",
                model=bert_model,
                tokenizer=bert_tokenizer
                # Consider adding device mapping if GPU is available: device=0 for CUDA
            )
            print("BERT QA pipeline loaded.")
        except Exception as e:
            print(
                f"Warning: Failed to load BERT QA pipeline '{bert_model_name}': {e}")
            print("Proceeding with LLM fallback only.")
            bert_qa_pipeline = None
    else:
        print("Warning: Transformers library not fully available. Cannot load BERT QA pipeline.")

    # --- Initialize LLM fallback ---
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3})  # Retrieve top 3 chunks
    llm = ChatOllama(model=llm_model_name)

    # Define LLM prompt for cases where BERT fails or isn't used
    llm_template = """
    Based ONLY on the following context, answer the question accurately and concisely.
    If the context does not contain the answer, say "Information not found".
    Do not add any extra explanation or commentary. Just provide the answer found in the context.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    llm_prompt = ChatPromptTemplate.from_template(llm_template)
    llm_chain = llm_prompt | llm | StrOutputParser()

    # --- Define processing functions ---
    def format_retriever_docs(docs):
        # Combine content of retrieved documents into a single context string
        return "\n\n".join(doc.page_content for doc in docs)

    def run_bert_qa(data):
        """Uses BERT pipeline to extract answers from context string"""
        if not bert_qa_pipeline:
            # print("DEBUG: BERT pipeline not available, skipping.")
            return None  # Skip if BERT couldn't load

        question = data.get("question")
        context = data.get("context_str")  # Use the combined context string
        if not question or not context:
            print("DEBUG: Missing question or context for BERT.")
            return None

        try:
            # print(f"DEBUG: Running BERT QA for: '{question}'")
            result = bert_qa_pipeline(
                question=question,
                context=context,
                max_answer_len=50,  # Limit answer length
                handle_impossible_answer=True  # Allow null answer if not found
            )
            # print(f"DEBUG: BERT Result: {result}")
            # Return answer only if score is above threshold and answer is not empty
            if result and result.get("score", 0) > BERT_SCORE_THRESHOLD and result.get("answer", "").strip():
                print(
                    f"DEBUG: BERT Answer found (Score: {result['score']:.2f}): '{result['answer']}'")
                return result["answer"].strip()
            else:
                # print("DEBUG: BERT answer below threshold or empty.")
                return None
        except Exception as e:
            print(f"Warning: BERT QA extraction error: {e}")
            return None

    def run_llm_fallback(data):
        """Runs LLM chain if BERT answer is None"""
        bert_answer = data.get("bert_answer")
        if bert_answer is not None:
            # print("DEBUG: Using BERT answer.")
            return bert_answer  # Return BERT answer if found and valid

        # print("DEBUG: BERT answer not found or invalid, running LLM fallback...")
        # Prepare input for the LLM chain (needs context_str and question)
        llm_input = {
            "context": data.get("context_str"),
            "question": data.get("question")
        }
        if not llm_input["context"] or not llm_input["question"]:
            print("DEBUG: Missing context or question for LLM fallback.")
            return "Error: Missing input for LLM."

        try:
            llm_answer = llm_chain.invoke(llm_input)
            print(f"DEBUG: LLM Fallback Answer: '{llm_answer}'")
            return llm_answer
        except Exception as e:
            print(f"Warning: LLM fallback invocation error: {e}")
            return "Error: LLM fallback failed."

    # --- Create the final chain ---
    # 1. Pass question through.
    # 2. Retrieve relevant documents.
    # 3. Format documents into a single context string.
    # 4. Run BERT QA on the context string.
    # 5. Run LLM fallback if BERT fails.
    # 6. Output the final answer string.
    _rag_chain = (
        {
            "question": RunnablePassthrough(),  # Pass the input question along
            "docs": RunnablePassthrough() | retriever  # Retrieve docs based on question
        }
        | RunnableLambda(lambda x: {
            "question": x["question"],
            # Combine docs into string
            "context_str": format_retriever_docs(x["docs"])
        })
        | RunnableLambda(lambda x: {
            "bert_answer": run_bert_qa(x),  # Run BERT (might return None)
            "context_str": x["context_str"],  # Pass context along
            "question": x["question"]  # Pass question along
        })
        | RunnableLambda(run_llm_fallback)  # Run LLM if bert_answer is None
    )
    print("Hybrid RAG chain created.")
    return _rag_chain

# --- NEW: Simple Function for Agent Tool ---


def get_info_from_rag(question: str) -> str:
    """
    Initializes RAG components (if needed) and answers a question.
    Designed to be used as a tool by an agent.
    """
    print(f"\n--- RAG Tool Invoked ---")
    print(f"DEBUG RAG Tool: Received question: '{question}'")
    global _rag_chain, _vector_store, _embeddings

    # Initialize components if they haven't been already
    if _embeddings is None:
        _embeddings = get_embedding_model()
    if _vector_store is None:
        _vector_store = load_or_create_vector_store(
            DOCS_PATH, FAISS_INDEX_PATH, _embeddings)
        if _vector_store is None:
            return "Error: Could not initialize knowledge base (vector store)."
    if _rag_chain is None:
        # Uses defaults defined above
        _rag_chain = create_rag_chain(_vector_store)
        if _rag_chain is None:
            return "Error: Could not create RAG chain."

    # Invoke the RAG chain
    try:
        print(f"DEBUG RAG Tool: Invoking chain for question: '{question}'")
        answer = _rag_chain.invoke(question)
        print(f"DEBUG RAG Tool: Chain returned answer: '{answer}'")
        # Basic check for common failure messages
        if not answer or "Error:" in answer or "Information not found" in answer:
            print(f"DEBUG RAG Tool: Answer indicates failure or not found.")
            # Return a clearer message for the agent if info isn't found
            return f"Information not found for '{question}'."
        return str(answer)  # Ensure it's a string
    except Exception as e:
        print(f"Error during RAG chain invocation in tool: {e}")
        # traceback.print_exc()
        return f"Error retrieving information for '{question}'."


# --- Main Execution (for testing RAG directly) ---
if __name__ == "__main__":
    print("--- RAG System Initialization (Direct Test) ---")
    # Test the tool function
    test_question = "What is my phone number?"  # Example question
    print(f"\nTesting RAG tool with question: '{test_question}'")
    answer = get_info_from_rag(test_question)
    print(f"\nResult from RAG tool:\n{answer}")

    test_question_2 = "Describe my role at Company X"  # Another example
    print(f"\nTesting RAG tool with question: '{test_question_2}'")
    answer_2 = get_info_from_rag(test_question_2)
    print(f"\nResult from RAG tool:\n{answer_2}")
