import logging
import os
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
from config import RAGConfig

logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self, config: RAGConfig):
        os.environ["FAISS_NO_GPU"] = "1"
        os.environ["FAISS_OPT_LEVEL"] = "generic"
        self.config = config
        self.embeddings = self._load_embeddings()
        self.vector_store = self._load_vector_store()
        self.qa_chain = self._create_chain()

    def _load_embeddings(self) -> Optional[HuggingFaceEmbeddings]:
        """Load sentence transformers embeddings"""
        try:
            logger.info(f"Loading embeddings: {self.config.embedding_model}")
            return HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                # Change to 'mps' for Apple Silicon
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.error(f"Embedding model failed: {str(e)}")
            return None

    def _load_vector_store(self) -> Optional[FAISS]:
        """Load or create FAISS vector store"""
        try:
            if os.path.exists(self.config.faiss_index_path):
                logger.info("Loading existing FAISS index")
                return FAISS.load_local(
                    self.config.faiss_index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            logger.warning(
                "No FAISS index found, create one with documents first")
            return None
        except Exception as e:
            logger.error(f"Vector store loading failed: {str(e)}")
            return None

    def _create_chain(self) -> Optional[RetrievalQA]:
        """Create RAG chain with proper library initialization"""
        if not self.vector_store:
            return None

        try:
            # Verify model file exists
            if not os.path.exists(self.config.llm_model):
                raise FileNotFoundError(
                    f"Model file not found: {self.config.llm_model}"
                )

            logger.info(f"Initializing LLM: {self.config.llm_model}")

            # Configure for Apple Silicon Metal acceleration
            llm = LlamaCpp(
                model_path=self.config.llm_model,
                n_gpu_layers=1,  # Use 1 for Metal acceleration
                n_ctx=4096,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                verbose=False,
                n_batch=512,  # Optimal for M1/M2
                n_threads=os.cpu_count() or 8,
            )

            return RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(),
                return_source_documents=True
            )
        except Exception as e:
            logger.error(f"Chain creation failed: {str(e)}")
            logger.debug("Common fixes:\n"
                         "1. Install required libraries: \n"
                         "   - brew install cmake\n"
                         "   - brew install libopenblas\n"
                         "2. For Apple Silicon, rebuild llama-cpp-python:\n"
                         "   CMAKE_ARGS='-DLLAMA_METAL=on' pip install --force-reinstall llama-cpp-python\n"
                         "3. Verify model file exists and is compatible")
            return None

    def query(self, question: str) -> Optional[str]:
        """Execute a RAG query with error handling"""
        if not self.qa_chain:
            return None

        try:
            result = self.qa_chain.invoke(question)
            return result.get("result", "")
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return None
