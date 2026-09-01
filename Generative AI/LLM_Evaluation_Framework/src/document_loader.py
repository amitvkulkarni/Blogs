"""
Document loading and ingestion for the RAG system.
Loads text files from data directory and chunks them for embedding.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import RAGConfig

# Suppress verbose logging
logging.getLogger('src.document_loader').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class DocumentIngester:
    """Handles loading and chunking documents for the RAG system."""
    
    def __init__(self, config: RAGConfig):
        """
        Initialize the document ingester.
        
        Args:
            config: RAGConfig object with chunking settings
        """
        self.config = config
        self.data_path = Path(config.indexing.data_path)
        
        # Initialize text splitter with config settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            separators=[config.chunking.separator, "\n", " ", ""],
            length_function=len,
        )
    
    def load_documents(self) -> List[Document]:
        """
        Load all text documents from the data directory.
        
        Returns:
            List[Document]: List of LangChain Document objects with metadata
            
        Raises:
            FileNotFoundError: If data directory doesn't exist
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        
        documents = []
        
        # Load all .txt files from the data directory
        txt_files = list(self.data_path.glob("*.txt"))
        
        if not txt_files:
            logger.warning(f"No text files found in {self.data_path}")
            return documents
        
        logger.info(f"Loading {len(txt_files)} text files from {self.data_path}")
        
        for file_path in txt_files:
            try:
                logger.info(f"Loading {file_path.name}...")
                documents.extend(self._load_single_file(file_path))
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
        
        logger.info(f"Loaded {len(documents)} total document chunks")
        return documents
    
    def _load_single_file(self, file_path: Path) -> List[Document]:
        """
        Load and chunk a single text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List[Document]: Chunked documents with metadata
        """
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into chunks
        chunks = self.text_splitter.split_text(content)
        
        # Create LangChain Document objects with metadata
        documents = []
        source_name = file_path.stem  # Filename without extension
        
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": str(file_path),
                    "source_name": source_name,
                    "chunk_index": i,
                    "file_name": file_path.name,
                    "total_chunks": len(chunks),
                }
            )
            documents.append(doc)
        
        logger.info(f"  {file_path.name}: {len(chunks)} chunks")
        return documents
    
    def get_chunk_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about the loaded and chunked documents.
        
        Args:
            documents: List of loaded documents
            
        Returns:
            Dict with statistics about the corpus
        """
        if not documents:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "avg_chunk_size": 0,
                "unique_sources": 0,
            }
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        sources = set(doc.metadata.get("source", "") for doc in documents)
        
        return {
            "total_chunks": len(documents),
            "total_characters": total_chars,
            "avg_chunk_size": total_chars // len(documents) if documents else 0,
            "unique_sources": len(sources),
            "sources": sorted(list(sources)),
        }


if __name__ == "__main__":
    # Test document ingestion
    logging.basicConfig(level=logging.INFO)
    
    from src.config import load_config
    
    try:
        config = load_config()
        ingester = DocumentIngester(config)
        documents = ingester.load_documents()
        stats = ingester.get_chunk_stats(documents)
        
        print("\n✓ Document ingestion successful")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Total characters: {stats['total_characters']}")
        print(f"  Average chunk size: {stats['avg_chunk_size']}")
        print(f"  Unique sources: {stats['unique_sources']}")
        print(f"  Sources: {stats['sources']}")
    except Exception as e:
        print(f"✗ Document ingestion failed: {e}")
