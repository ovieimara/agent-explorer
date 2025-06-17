from operator import itemgetter
import os
from tracemalloc import start
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from transformers import BertForQuestionAnswering, BertTokenizerFast, pipeline
import torch
from langchain.schema.runnable import RunnableLambda
import time

# --- Configuration ---
DOCS_PATH = "docs"
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Good general-purpose, runs locally
LLAMA3_MODEL = "llama3:8b"
DEEPSEEK_MODEL = "deepseek-r1:8b"  # Adjust if you pulled a different tag
MISTRAL_MODEL = "mistral:7b-instruct"  # Adjust if you pulled a different tag
GEMMA_MODEL = "gemma3:12b-it-qat"  # Adjust if you pulled a different tag
QWEN_MODEL = "qwen2:7b"

# --- Functions ---


def load_and_chunk_docs(directory_path):
    """Loads PDF documents from a directory and splits them into chunks."""
    print(f"Loading documents from: {directory_path}")
    # Using PyPDFLoader for PDF files
    loader = DirectoryLoader(directory_path, glob="**/*.pdf",
                             loader_cls=PyPDFLoader, recursive=True, show_progress=True)
    documents = loader.load()
    if not documents:
        print("No documents found. Please add PDF files to the 'docs' directory.")
        return None

    print(f"Loaded {len(documents)} document pages.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks


def create_or_load_vector_store(chunks, embedding_model, index_path):
    """Creates a FAISS vector store if it doesn't exist, otherwise loads it."""
    if os.path.exists(index_path):
        print(f"Loading existing FAISS index from: {index_path}")
        # M1/M2 needs this flag sometimes
        vector_store = FAISS.load_local(
            index_path, embedding_model, allow_dangerous_deserialization=True)
        print("FAISS index loaded.")
    else:
        if not chunks:
            print("Cannot create vector store without document chunks.")
            return None
        print("Creating new FAISS index...")
        vector_store = FAISS.from_documents(chunks, embedding_model)
        print("FAISS index created.")
        print(f"Saving FAISS index to: {index_path}")
        vector_store.save_local(index_path)
        print("FAISS index saved.")
    return vector_store


def create_rag_chain(vector_store, llm_model, bert_model_name="bert-large-uncased-whole-word-masking-finetuned-squad"):
    try:
        # Load model and tokenizer together to ensure compatibility
        model = BertForQuestionAnswering.from_pretrained(bert_model_name)
        tokenizer = BertTokenizerFast.from_pretrained(bert_model_name)

        # Initialize pipeline with explicit model and tokenizer
        bert_qa = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        print(f"Failed to initialize BERT: {str(e)}")
        return None

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model=llm_model, temperature=0, max_tokens=100)

    template = """Extract the exact value from context. Return ONLY the value. Context: {context}"""
    prompt = ChatPromptTemplate.from_template(template)

    def bert_extractor(data):
        try:
            if not data["context"].strip():
                return None
            question = f"what is {data['question']}?"
            result = bert_qa(
                question=question,
                context=data["context"],
                max_answer_len=100,
                handle_impossible_answer=False  # Get best guess anyway
            )

            # Handle different return formats
            answer = result.get("answer", "").strip()
            score = result.get("score", 0)

            if answer and score > 0.1:  # Lower threshold
                print(f"BERT success: {answer[:50]} (score: {score:.2f})")
                return answer
            return None
        except Exception as e:
            print(f"BERT failed: {str(e)}")
            return None

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | RunnableLambda(lambda x: {
            "context": " ".join([doc.page_content for doc in x["context"]]),
            "question": x["question"]
        })
        | {
            "bert_answer": RunnableLambda(bert_extractor),
            "question": itemgetter("question"),
            "context": itemgetter("context")
        }
        | {
            "answer": lambda x: (
                x["bert_answer"] or
                llm.invoke(prompt.format(
                    question=x["question"],
                    context=x["context"]
                )).content.strip()
            ),
            "source": lambda x: "BERT" if x["bert_answer"] else "LLM"
        }
    )

    return rag_chain


def get_embedding_model(model_name):
    return HuggingFaceEmbeddings(model_name=model_name)


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Initialize Embedding Model (runs locally)
    print("Initializing embedding model...")
    embeddings = get_embedding_model(model_name=EMBEDDING_MODEL_NAME)
    print(f"Using embedding model: {EMBEDDING_MODEL_NAME}")

    # 2. Load or Create FAISS Vector Store
    # Check if index exists before potentially loading docs (saves time if index is present)
    if not os.path.exists(FAISS_INDEX_PATH):
        doc_chunks = load_and_chunk_docs(DOCS_PATH)
    else:
        doc_chunks = None  # Don't need chunks if loading index

    vector_store = create_or_load_vector_store(
        doc_chunks, embeddings, FAISS_INDEX_PATH)

    if vector_store:
        # 3. Create RAG chains for each LLM
        print("\nCreating RAG chain for Llama 3...")
        rag_chain_llama3 = create_rag_chain(vector_store, LLAMA3_MODEL)

        print("Creating RAG chain for DeepSeek...")
        rag_chain_deepseek = create_rag_chain(vector_store, DEEPSEEK_MODEL)

        print("Creating RAG chain for Mistral...")
        rag_chain_mistral = create_rag_chain(vector_store, MISTRAL_MODEL)

        print("Creating RAG chain for Gemma...")
        rag_chain_gemma = create_rag_chain(vector_store, GEMMA_MODEL)

        print("Creating RAG chain for Qwen...")
        rag_chain_qwen = create_rag_chain(vector_store, QWEN_MODEL)

        # 4. Ask Questions!
        print("\n--- Ready to Query ---")
        while True:
            question = input(
                "Ask a question about your documents (or type 'quit' to exit): ")
            if question.lower() == 'quit':
                break
            if not question:
                continue

            if rag_chain_llama3:
                begin = time.time()
                print(f"\nAsking Llama 3 ({LLAMA3_MODEL})...")
                answer_llama3 = rag_chain_llama3.invoke(question)
                print(f"Llama 3 Answer:\n{answer_llama3}")
                end = time.time()
                print(f"run time for llama is {end - begin:.2f} secs...")

            if rag_chain_deepseek:
                begin = time.time()
                print(f"\nAsking DeepSeek Coder ({DEEPSEEK_MODEL})...")
                # Note: DeepSeek Coder is primarily for code, its general Q&A might be limited
                answer_deepseek = rag_chain_deepseek.invoke(question)
                print(f"DeepSeek R1 Answer:\n{answer_deepseek}")
                end = time.time()
                print(f"run time for deepseek is {end - begin:.2f} secs...")

            if rag_chain_mistral:
                begin = time.time()
                print(f"\nAsking Mistral ({MISTRAL_MODEL})...")
                answer_mistral = rag_chain_mistral.invoke(question)
                print(f"Mistral 3 Answer:\n{answer_mistral}")
                end = time.time()
                print(f"run time for Mistral is {end - begin:.2f} secs...")

            # if rag_chain_gemma:
            #     begin = time.time()
            #     print(f"\nAsking Gemma ({GEMMA_MODEL})...")
            #     answer_gemma = rag_chain_gemma.invoke(question)
            #     print(f"Gemma 3 Answer:\n{answer_gemma}")
            #     end = time.time()
            #     print(f"run time for Gemma is {end - begin:.2f} secs...")

            if rag_chain_qwen:
                begin = time.time()
                print(f"\nAsking Qwen ({QWEN_MODEL})...")
                answer_qwen = rag_chain_qwen.invoke(question)
                print(f"Qwen 2.5 Answer:\n{answer_qwen}")
                end = time.time()
                print(f"run time for Qwen is {end - begin:.2f} secs...")
    else:
        print("\nFailed to initialize RAG system. Exiting.")
