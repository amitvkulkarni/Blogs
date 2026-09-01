"""
RAG pipeline implementation using LangChain.
Orchestrates retrieval and generation with Azure OpenAI models.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.config import RAGConfig, ModelConfig

# Suppress verbose logging
logging.getLogger('src.rag_pipeline').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class RAGPipeline:
    """Implements a RAG pipeline with retrieval-augmented generation."""
    
    def __init__(self, config: RAGConfig, retriever: BaseRetriever):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: RAGConfig object
            retriever: LangChain retriever for document retrieval
        """
        self.config = config
        self.retriever = retriever
        self.models: Dict[str, AzureChatOpenAI] = {}
        self.chains: Dict[str, Any] = {}
        
        # Initialize Azure OpenAI models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize Azure OpenAI chat models from config."""
        for model_config in self.config.models:
            try:
                llm = AzureChatOpenAI(
                    azure_deployment=model_config.deployment_name,
                    model=model_config.model_id,
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_tokens,
                    api_key=self.config.azure_api_key,
                    azure_endpoint=self.config.azure_api_endpoint,
                    api_version="2024-12-01-preview",
                )
                self.models[model_config.name] = llm
                logger.info(f"Initialized model: {model_config.name} ({model_config.deployment_name}")
            except Exception as e:
                logger.error(f"Failed to initialize model {model_config.name}: {e}")
    
    def _format_context(self, docs: List[Document]) -> str:
        """
        Format retrieved documents as context for the prompt.
        
        Args:
            docs: List of retrieved documents
            
        Returns:
            str: Formatted context string with citations
        """
        if not docs:
            return "No relevant documents found."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_name", "Unknown")
            content = doc.page_content.strip()
            formatted.append(f"[Source {i}: {source}]\n{content}")
        
        return "\n\n".join(formatted)
    
    def _create_rag_chain(self, model_name: str):
        """
        Create a RAG chain for a specific model.
        
        Args:
            model_name: Name of the model from config
            
        Returns:
            Chain: LangChain chain for RAG
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in initialized models")
        
        llm = self.models[model_name]
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.config.prompts.system_message),
            ("human", self.config.prompts.user_template),
        ])
        
        # Create RAG chain: retrieve docs -> format context -> pass to LLM
        def retrieve_and_format(inputs):
            query = inputs.get("question", "")
            docs = self.retriever.invoke(query)
            context = self._format_context(docs)
            return {
                "context": context,
                "question": query,
                "retrieved_docs": docs,
            }
        
        # Chain: retrieval -> formatting -> LLM generation
        chain = (
            RunnablePassthrough.assign(
                context=lambda x: self._format_context(
                    self.retriever.invoke(x.get("question", ""))
                ),
                retrieved_docs=lambda x: self.retriever.invoke(
                    x.get("question", "")
                ),
            )
            | prompt_template
            | llm
            | StrOutputParser()
        )
        
        return chain, retrieve_and_format
    
    def query(
        self,
        question: str,
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Execute a RAG query with a specific model.
        
        Args:
            question: The user's question
            model_name: Name of the model to use
            
        Returns:
            Dict with answer, retrieved documents, and citations
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        logger.info(f"Query with model '{model_name}': {question[:100]}...")
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.invoke(question)
        context = self._format_context(retrieved_docs)
        
        # Get the model
        llm = self.models[model_name]
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.config.prompts.system_message),
            ("human", self.config.prompts.user_template),
        ])
        
        # Create chain and run
        chain = prompt_template | llm | StrOutputParser()
        
        answer = chain.invoke({
            "context": context,
            "question": question,
        })
        
        # Extract sources from retrieved docs
        sources = [
            {
                "source_name": doc.metadata.get("source_name", "Unknown"),
                "file_name": doc.metadata.get("file_name", "Unknown"),
                "chunk_index": doc.metadata.get("chunk_index", 0),
            }
            for doc in retrieved_docs
        ]
        
        result = {
            "question": question,
            "answer": answer,
            "model_name": model_name,
            "retrieved_documents": retrieved_docs,
            "sources": sources,
            "context_snippet": context[:500] + "..." if len(context) > 500 else context,
        }
        
        logger.info(f"Generated answer with {len(retrieved_docs)} retrieved documents")
        return result
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available model names.
        
        Returns:
            List[str]: Model names configured for the pipeline
        """
        return list(self.models.keys())


if __name__ == "__main__":
    # Test RAG pipeline
    logging.basicConfig(level=logging.INFO)
    
    from src.config import load_config
    from src.document_loader import DocumentIngester
    from src.vector_store import VectorStoreManager
    
    try:
        config = load_config()
        
        # Load documents and build index
        ingester = DocumentIngester(config)
        documents = ingester.load_documents()
        
        manager = VectorStoreManager(config)
        manager.build_index(documents)
        retriever = manager.get_retriever()
        
        # Create RAG pipeline
        pipeline = RAGPipeline(config, retriever)
        
        # Test query
        test_question = "What are the HVAC efficiency requirements for Chicago?"
        available_models = pipeline.get_available_models()
        
        print(f"\n✓ RAG pipeline initialized")
        print(f"  Available models: {available_models}")
        
        if available_models:
            result = pipeline.query(test_question, available_models[0])
            print(f"\n  Question: {result['question']}")
            print(f"  Model: {result['model_name']}")
            print(f"  Retrieved {len(result['retrieved_documents'])} documents")
            print(f"  Answer preview: {result['answer'][:200]}...")
    except Exception as e:
        print(f"✗ RAG pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
