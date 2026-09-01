"""
Vector store management for the RAG system.
Handles embedding documents and storing them in a persistent Chroma database.
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from src.config import RAGConfig

# Suppress verbose logging
logging.getLogger('src.vector_store').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class VectorStoreManager:
    """Manages the persistent vector store and retrieval."""
    
    def __init__(self, config: RAGConfig):
        """
        Initialize the vector store manager.
        
        Args:
            config: RAGConfig object with vector store and Azure settings
        """
        self.config = config
        self.persist_directory = Path(config.vector_store.path)
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        
        # Initialize embeddings from Azure OpenAI
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize OpenAI embeddings from Azure."""
        if not self.config.azure_api_key or not self.config.azure_api_endpoint:
            raise ValueError("Azure OpenAI credentials not configured")
        
        self.embeddings = AzureOpenAIEmbeddings(
            deployment=self.config.embedding.deployment_name,
            model=self.config.embedding.model_id,
            api_key=self.config.azure_api_key,
            azure_endpoint=self.config.azure_api_endpoint,
            api_version="2024-06-01",
        )
        logger.info(f"Initialized embeddings with deployment: {self.config.embedding.deployment_name}")
    
    def build_index(self, documents: List[Document], force_rebuild: bool = False) -> Chroma:
        """
        Build or rebuild the vector store from documents.
        
        Args:
            documents: List of LangChain Document objects to embed
            force_rebuild: If True, rebuild even if index exists
            
        Returns:
            Chroma: The initialized vector store
        """
        persist_dir = Path(self.persist_directory)
        
        # Check if index already exists and has documents
        if persist_dir.exists() and not force_rebuild:
            logger.info(f"Vector store already exists at {persist_dir}, checking if populated...")
            try:
                # Load and check if collection has documents
                temp_store = Chroma(
                    collection_name=self.config.vector_store.collection_name,
                    persist_directory=str(persist_dir),
                    embedding_function=self.embeddings,
                )
                doc_count = temp_store._collection.count() if hasattr(temp_store._collection, 'count') else 0
                
                if doc_count > 0:
                    logger.info(f"Vector store has {doc_count} documents, loading existing index...")
                    return self.load_index()
                else:
                    logger.info(f"Vector store exists but is empty ({doc_count} documents), rebuilding...")
                    shutil.rmtree(persist_dir)
            except Exception as e:
                logger.warning(f"Could not check existing index: {e}, rebuilding...")
                if persist_dir.exists():
                    shutil.rmtree(persist_dir)
        
        # Remove existing index if force rebuild
        if persist_dir.exists() and force_rebuild:
            logger.info(f"Removing existing vector store for rebuild...")
            shutil.rmtree(persist_dir)
        
        # Create directory
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Building vector store with {len(documents)} documents...")
        
        # Create the vector store by embedding documents
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.config.vector_store.collection_name,
            persist_directory=str(persist_dir),
        )
        
        # Persist the store
        self.vector_store.persist()
        logger.info(f"Vector store built and persisted to {persist_dir}")
        
        # Create retriever
        self._create_retriever()
        
        return self.vector_store
    
    def load_index(self) -> Chroma:
        """
        Load an existing vector store from disk.
        
        Returns:
            Chroma: The loaded vector store
            
        Raises:
            FileNotFoundError: If vector store doesn't exist
        """
        persist_dir = Path(self.persist_directory)
        
        if not persist_dir.exists():
            raise FileNotFoundError(f"Vector store not found at {persist_dir}")
        
        logger.info(f"Loading vector store from {persist_dir}...")
        
        self.vector_store = Chroma(
            collection_name=self.config.vector_store.collection_name,
            persist_directory=str(persist_dir),
            embedding_function=self.embeddings,
        )
        
        # Create retriever
        self._create_retriever()
        
        logger.info("Vector store loaded successfully")
        return self.vector_store
    
    def _create_retriever(self):
        """Create a retriever from the vector store."""
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized")
        
        self.retriever = self.vector_store.as_retriever(
            search_type=self.config.vector_store.search_type,
            search_kwargs={"k": self.config.vector_store.k_retrieval}
        )
        logger.info(f"Created retriever with k={self.config.vector_store.k_retrieval}")
    
    def get_retriever(self) -> BaseRetriever:
        """
        Get the retriever for querying the vector store.
        
        Returns:
            BaseRetriever: LangChain retriever for document retrieval
            
        Raises:
            RuntimeError: If retriever not initialized
        """
        if self.retriever is None:
            raise RuntimeError("Retriever not initialized. Call build_index() or load_index() first.")
        return self.retriever
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            k: Number of results (defaults to config k_retrieval)
            
        Returns:
            List[Tuple[Document, float]]: Retrieved documents with relevance scores
        """
        if self.retriever is None:
            raise RuntimeError("Retriever not initialized")
        
        # Use retriever to get documents (invoke for LangChain 1.x)
        documents = self.retriever.invoke(query)
        
        # Return documents (Chroma retriever may or may not have scores)
        # For similarity search, we can get scores using similarity_search_with_score
        if self.config.vector_store.search_type == "similarity":
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k or self.config.vector_store.k_retrieval
            )
            return results
        else:
            # For MMR or other search types, return documents without scores
            return [(doc, 0.0) for doc in documents]
    
    def get_index_stats(self) -> dict:
        """
        Get statistics about the vector store.
        
        Returns:
            dict: Statistics including number of documents
        """
        if self.vector_store is None:
            return {"status": "not_initialized"}
        
        try:
            # Get collection info
            collection = self.vector_store._collection
            return {
                "status": "initialized",
                "collection_name": self.config.vector_store.collection_name,
                "persist_path": str(self.persist_directory),
                "document_count": collection.count() if hasattr(collection, 'count') else "unknown",
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    # Test vector store
    logging.basicConfig(level=logging.INFO)
    
    from src.config import load_config
    from src.document_loader import DocumentIngester
    
    try:
        config = load_config()
        
        # Load documents
        ingester = DocumentIngester(config)
        documents = ingester.load_documents()
        
        # Build vector store
        manager = VectorStoreManager(config)
        manager.build_index(documents)
        
        # Test retrieval
        test_query = "What are the HVAC efficiency requirements for Chicago?"
        results = manager.retrieve(test_query, k=3)
        
        print("\n✓ Vector store test successful")
        print(f"  Retrieved {len(results)} documents")
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. Score: {score:.3f}")
            print(f"     Source: {doc.metadata.get('file_name')}")
            print(f"     Preview: {doc.page_content[:100]}...")
    except Exception as e:
        print(f"✗ Vector store test failed: {e}")
        import traceback
        traceback.print_exc()
