"""
Configuration loader for RAG system.
Loads environment variables from .env and settings from config.yaml.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration for a single language model."""
    name: str
    deployment_name: str
    model_id: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1500, gt=0)


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding model."""
    deployment_name: str
    model_id: str
    embedding_dimension: int = Field(default=1536, gt=0)


class VectorStoreConfig(BaseModel):
    """Configuration for the vector store."""
    path: str = Field(default="./vector_store")
    collection_name: str = Field(default="engineering_docs")
    search_type: str = Field(default="similarity")
    k_retrieval: int = Field(default=4, gt=0)


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    separator: str = Field(default="\n\n")


class PromptsConfig(BaseModel):
    """Configuration for RAG prompts."""
    system_message: str
    user_template: str


class EvaluationThresholds(BaseModel):
    """Thresholds for evaluation metrics."""
    faithfulness: float = Field(default=0.3, ge=0.0, le=1.0)
    answer_relevancy: float = Field(default=0.3, ge=0.0, le=1.0)
    correctness: float = Field(default=0.3, ge=0.0, le=1.0)


class EvaluationConfig(BaseModel):
    """Configuration for evaluation."""
    golden_dataset_path: str = Field(default="./golden_dataset/qa_pairs.json")
    sample_limit: Optional[int] = Field(default=None)
    evaluator_model: str = Field(default="gpt5-4", description="Model to use for evaluation")
    metrics: List[str] = Field(default=["faithfulness", "answer_relevancy", "correctness"])
    thresholds: EvaluationThresholds = Field(default_factory=EvaluationThresholds)


class IndexingConfig(BaseModel):
    """Configuration for indexing."""
    rebuild_index: bool = Field(default=False)
    data_path: str = Field(default="./data")


class RAGConfig(BaseModel):
    """Complete configuration for the RAG system."""
    chunking: ChunkingConfig
    vector_store: VectorStoreConfig
    models: List[ModelConfig]
    embedding: EmbeddingConfig
    prompts: PromptsConfig
    evaluation: EvaluationConfig
    indexing: IndexingConfig
    logging_level: str = Field(default="INFO")
    logging_console: bool = Field(default=True)

    # Azure credentials from environment
    azure_api_key: Optional[str] = None
    azure_api_endpoint: Optional[str] = None
    
    @validator('models')
    def validate_models(cls, v):
        if not v or len(v) < 2:
            raise ValueError("At least two models must be configured for comparison")
        return v


class ConfigLoader:
    """Loads and manages RAG system configuration."""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Initialize the configuration loader.
        
        Args:
            config_file: Path to the YAML configuration file
        """
        self.config_file = Path(config_file)
        self.config: Optional[RAGConfig] = None
        
    def load(self) -> RAGConfig:
        """
        Load configuration from .env and config.yaml files.
        
        Returns:
            RAGConfig: Validated configuration object
            
        Raises:
            FileNotFoundError: If config.yaml is not found
            ValueError: If configuration is invalid
        """
        # Load environment variables from .env
        load_dotenv()
        
        # Load YAML configuration
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Extract and validate YAML sections
        config_dict = {
            'chunking': yaml_data.get('chunking', {}),
            'vector_store': yaml_data.get('vector_store', {}),
            'models': yaml_data.get('models', []),
            'embedding': yaml_data.get('embedding', {}),
            'prompts': yaml_data.get('prompts', {}),
            'evaluation': yaml_data.get('evaluation', {}),
            'indexing': yaml_data.get('indexing', {}),
            'logging_level': yaml_data.get('logging', {}).get('level', 'INFO'),
            'logging_console': yaml_data.get('logging', {}).get('console_output', True),
        }
        
        # Load Azure credentials from environment
        config_dict['azure_api_key'] = os.getenv('AZURE_OPENAI_API_KEY')
        config_dict['azure_api_endpoint'] = os.getenv('AZURE_OPENAI_ENDPOINT')
        
        # Validate that Azure credentials are set
        if not config_dict['azure_api_key']:
            logger.warning("AZURE_OPENAI_API_KEY not found in .env file")
        if not config_dict['azure_api_endpoint']:
            logger.warning("AZURE_OPENAI_ENDPOINT not found in .env file")
        
        # Create and validate config
        try:
            self.config = RAGConfig(**config_dict)
            logger.info(f"Configuration loaded successfully from {self.config_file}")
            logger.info(f"Configured {len(self.config.models)} models for comparison")
            return self.config
        except Exception as e:
            raise ValueError(f"Failed to validate configuration: {str(e)}")
    
    def get(self) -> RAGConfig:
        """
        Get the loaded configuration.
        
        Returns:
            RAGConfig: The configuration object
            
        Raises:
            RuntimeError: If configuration has not been loaded
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")
        return self.config
    
    def validate_azure_credentials(self) -> bool:
        """
        Validate that Azure credentials are available.
        
        Returns:
            bool: True if credentials are valid, False otherwise
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load() first.")
        
        if not self.config.azure_api_key or not self.config.azure_api_endpoint:
            logger.error("Azure credentials not configured. Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in .env")
            return False
        
        return True


def load_config(config_file: str = "config.yaml") -> RAGConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_file: Path to the YAML configuration file
        
    Returns:
        RAGConfig: Validated configuration object
    """
    loader = ConfigLoader(config_file)
    return loader.load()


if __name__ == "__main__":
    # Test configuration loading
    logging.basicConfig(level=logging.INFO)
    try:
        config = load_config()
        print("✓ Configuration loaded successfully")
        print(f"  Models: {[m.name for m in config.models]}")
        print(f"  Embedding: {config.embedding.model_id}")
        print(f"  Vector store: {config.vector_store.path}")
        print(f"  Chunk size: {config.chunking.chunk_size}")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
