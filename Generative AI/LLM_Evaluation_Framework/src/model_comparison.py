"""
Model comparison runner for evaluating multiple models on the same queries.
Provides normalized output for DeepEval evaluation.
"""

import logging
from typing import Dict, List, Any, Tuple
from src.config import RAGConfig
from src.rag_pipeline import RAGPipeline
from langchain_core.retrievers import BaseRetriever

# Suppress verbose logging
logging.getLogger('src.model_comparison').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class ModelComparator:
    """Compares multiple models on the same RAG queries."""
    
    def __init__(self, config: RAGConfig, retriever: BaseRetriever):
        """
        Initialize the model comparator.
        
        Args:
            config: RAGConfig object
            retriever: LangChain retriever
        """
        self.config = config
        self.pipeline = RAGPipeline(config, retriever)
        self.results: List[Dict[str, Any]] = []
    
    def compare_models(
        self,
        question: str,
        use_models: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the same question through multiple models and compare results.
        
        Args:
            question: The question to ask
            use_models: Specific models to use (None = all configured models)
            
        Returns:
            Dict containing results from all models for comparison
        """
        available_models = self.pipeline.get_available_models()
        
        if use_models is None:
            use_models = available_models
        else:
            # Validate requested models exist
            invalid = [m for m in use_models if m not in available_models]
            if invalid:
                raise ValueError(f"Invalid models: {invalid}")
        
        if len(use_models) < 2:
            logger.warning("Comparison requires at least 2 models")
        
        logger.info(f"Comparing {len(use_models)} models on question: {question[:80]}...")
        
        # Run query with each model
        model_results = {}
        for model_name in use_models:
            try:
                result = self.pipeline.query(question, model_name)
                model_results[model_name] = result
            except Exception as e:
                logger.error(f"Error querying model {model_name}: {e}")
                model_results[model_name] = {
                    "question": question,
                    "model_name": model_name,
                    "answer": f"ERROR: {str(e)}",
                    "error": str(e),
                    "retrieved_documents": [],
                    "sources": [],
                }
        
        # Normalize and compare results
        comparison_result = {
            "question": question,
            "model_count": len(model_results),
            "models": model_results,
            "comparison_summary": self._generate_summary(model_results),
        }
        
        self.results.append(comparison_result)
        return comparison_result
    
    def _generate_summary(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comparison summary across models.
        
        Args:
            model_results: Results from all models
            
        Returns:
            Dict with comparison metrics
        """
        summary = {
            "models_compared": list(model_results.keys()),
            "all_sources": self._extract_all_sources(model_results),
            "model_details": [],
        }
        
        for model_name, result in model_results.items():
            if "error" not in result:
                summary["model_details"].append({
                    "model_name": model_name,
                    "answer_length": len(result.get("answer", "")),
                    "doc_count": len(result.get("retrieved_documents", [])),
                    "has_sources": len(result.get("sources", [])) > 0,
                })
        
        return summary
    
    def _extract_all_sources(self, model_results: Dict[str, Any]) -> List[str]:
        """
        Extract all unique sources used across all model results.
        
        Args:
            model_results: Results from all models
            
        Returns:
            List of unique source names
        """
        sources = set()
        for result in model_results.values():
            for source in result.get("sources", []):
                sources.add(source.get("source_name", "Unknown"))
        return sorted(list(sources))
    
    def batch_compare(
        self,
        questions: List[str],
        use_models: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compare models on a batch of questions.
        
        Args:
            questions: List of questions to compare
            use_models: Specific models to use (None = all)
            
        Returns:
            List of comparison results
        """
        logger.info(f"Running batch comparison on {len(questions)} questions with {len(use_models or self.pipeline.get_available_models())} models")
        
        batch_results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"  [{i}/{len(questions)}] Processing question...")
            try:
                result = self.compare_models(question, use_models)
                batch_results.append(result)
            except Exception as e:
                logger.error(f"Error on question {i}: {e}")
                batch_results.append({
                    "question": question,
                    "error": str(e),
                    "models": {},
                })
        
        logger.info(f"Batch comparison complete. {len(batch_results)} results.")
        return batch_results
    
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get all comparison results from this session.
        
        Returns:
            List of all comparison results
        """
        return self.results
    
    def format_comparison_for_eval(self, comparison_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a comparison result for DeepEval evaluation.
        
        Args:
            comparison_result: A single comparison result
            
        Returns:
            Dict formatted for evaluation
        """
        eval_ready = {
            "question": comparison_result["question"],
            "model_responses": {},
        }
        
        for model_name, result in comparison_result["models"].items():
            if "error" not in result:
                eval_ready["model_responses"][model_name] = {
                    "answer": result["answer"],
                    "context": result.get("context_snippet", ""),
                    "sources": result.get("sources", []),
                }
        
        return eval_ready


if __name__ == "__main__":
    # Test model comparison
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
        
        # Create comparator
        comparator = ModelComparator(config, retriever)
        
        # Test comparison
        test_questions = [
            "What are the HVAC efficiency requirements for Chicago?",
            "What seismic design category is Seattle in?",
        ]
        
        print("\n✓ Model comparison initialized")
        results = comparator.batch_compare(test_questions)
        
        print(f"\n✓ Comparison complete: {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"\n  Question {i}: {result['question'][:50]}...")
            for model_name in result.get("models", {}).keys():
                if "error" not in result["models"][model_name]:
                    print(f"    {model_name}: OK")
                else:
                    print(f"    {model_name}: ERROR")
    except Exception as e:
        print(f"✗ Model comparison test failed: {e}")
        import traceback
        traceback.print_exc()
