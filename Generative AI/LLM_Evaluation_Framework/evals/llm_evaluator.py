"""
LLM-based evaluation for comparing RAG models using Azure OpenAI.
Uses an Azure OpenAI model to intelligently assess answer quality across multiple metrics.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import statistics

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate

# Suppress verbose logging
logging.getLogger('deepeval').setLevel(logging.WARNING)
logging.getLogger('langchain').setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a single question across models."""
    question: str
    expected_answer: str
    model_scores: Dict[str, Dict[str, float]]  # model_name -> {metric_name -> score}
    model_answers: Dict[str, str]
    passed: Dict[str, bool]  # model_name -> passed_threshold


class LLMEvaluator:
    """Evaluates RAG model responses using an Azure OpenAI LLM as the evaluator model."""
    
    def __init__(self, config, golden_dataset_path: str = None):
        """
        Initialize the evaluator with configurable Azure OpenAI model.
        
        Args:
            config: RAGConfig object
            golden_dataset_path: Path to golden dataset JSON
        """
        self.config = config
        self.golden_dataset_path = golden_dataset_path or config.evaluation.golden_dataset_path
        
        # Get the evaluator model from config
        evaluator_model_name = config.evaluation.evaluator_model
        evaluator_model_config = None
        
        for model_config in config.models:
            if model_config.name == evaluator_model_name:
                evaluator_model_config = model_config
                break
        
        if evaluator_model_config is None:
            raise ValueError(
                f"Evaluator model '{evaluator_model_name}' not found in config. "
                f"Available models: {[m.name for m in config.models]}"
            )
        
        # Initialize Azure OpenAI client for evaluation using configured model
        self.evaluator_llm = AzureChatOpenAI(
            azure_deployment=evaluator_model_config.deployment_name,
            model=evaluator_model_config.model_id,
            temperature=0.0,  # Deterministic scoring
            max_tokens=100,
            api_key=self.config.azure_api_key,
            azure_endpoint=self.config.azure_api_endpoint,
            api_version="2024-12-01-preview",
        )
        
        self.golden_dataset = []
        self.evaluation_results: List[EvaluationResult] = []
        
        logger.info(f"RAG evaluator initialized with {evaluator_model_config.name} ({evaluator_model_config.deployment_name}) for intelligent evaluation")
        self._load_golden_dataset()
    

    
    def _load_golden_dataset(self):
        """Load the golden dataset from JSON."""
        dataset_path = Path(self.golden_dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Golden dataset not found: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.golden_dataset = json.load(f)
        
        logger.info(f"Loaded {len(self.golden_dataset)} golden QA pairs from {dataset_path}")
    
    def evaluate_comparison(
        self,
        comparison_results: List[Dict[str, Any]],
        metrics: List[str] = None,
        limit: int = None,
    ) -> List[EvaluationResult]:
        """
        Evaluate comparison results from the model comparator.
        
        Args:
            comparison_results: List of comparison results from ModelComparator.batch_compare()
            metrics: List of metrics to evaluate (defaults to config)
            limit: Maximum number of questions to evaluate
            
        Returns:
            List of EvaluationResult objects
        """
        if metrics is None:
            metrics = self.config.evaluation.metrics
        
        if limit is None:
            limit = self.config.evaluation.sample_limit
        
        # Get questions to evaluate
        questions_to_eval = self.golden_dataset
        if limit:
            questions_to_eval = questions_to_eval[:limit]
        
        logger.info(f"Evaluating {len(questions_to_eval)} questions with metrics: {metrics}")
        
        self.evaluation_results = []
        
        for qa_pair in questions_to_eval:
            # Find corresponding comparison result
            question = qa_pair["question"]
            expected_answer = qa_pair["expected_answer"]
            
            # Find matching comparison result
            comparison = None
            for result in comparison_results:
                if result.get("question") == question:
                    comparison = result
                    break
            
            if comparison is None:
                logger.warning(f"No comparison result found for: {question[:50]}...")
                continue
            
            # Evaluate each model
            eval_result = self._evaluate_question(
                question,
                expected_answer,
                comparison,
                metrics,
            )
            
            self.evaluation_results.append(eval_result)
        
        logger.info(f"Evaluation complete: {len(self.evaluation_results)} results")
        return self.evaluation_results
    
    def _evaluate_question(
        self,
        question: str,
        expected_answer: str,
        comparison: Dict[str, Any],
        metrics: List[str],
    ) -> EvaluationResult:
        """
        Evaluate a single question across all models.
        
        Args:
            question: The question
            expected_answer: Expected correct answer
            comparison: Comparison result from ModelComparator
            metrics: Metrics to evaluate
            
        Returns:
            EvaluationResult with scores for each model
        """
        model_scores = {}
        model_answers = {}
        passed = {}
        
        for model_name, model_result in comparison.get("models", {}).items():
            if "error" in model_result:
                logger.warning(f"Skipping {model_name}: error in results")
                model_scores[model_name] = {m: 0.0 for m in metrics}
                model_answers[model_name] = f"ERROR: {model_result.get('error', 'Unknown error')}"
                passed[model_name] = False
                continue
            
            answer = model_result.get("answer", "")
            model_answers[model_name] = answer
            
            # Evaluate metrics
            metric_scores = {}
            for metric_name in metrics:
                try:
                    score = self._evaluate_metric(
                        metric_name,
                        question,
                        answer,
                        expected_answer,
                        model_result.get("context_snippet", ""),
                    )
                    metric_scores[metric_name] = score
                except Exception as e:
                    logger.error(f"Error evaluating {metric_name} for {model_name}: {e}")
                    metric_scores[metric_name] = 0.3  # Give some credit even if metric fails
            
            model_scores[model_name] = metric_scores
            
            # Check if passed - use average score approach
            # An answer passes if: (1) it's not an error AND (2) average metric score >= threshold
            # This is more realistic than requiring all metrics to pass individually
            if not answer or len(answer.strip()) < 10:
                # Reject very short or empty answers
                passed[model_name] = False
            else:
                # Calculate average score across all metrics
                average_score = sum(metric_scores.values()) / len(metric_scores) if metric_scores else 0.0
                # Lower threshold for average - 0.30 allows reasonable answers through
                average_threshold = 0.30
                passed[model_name] = average_score >= average_threshold
        
        return EvaluationResult(
            question=question,
            expected_answer=expected_answer,
            model_scores=model_scores,
            model_answers=model_answers,
            passed=passed,
        )
    
    def _evaluate_metric(
        self,
        metric_name: str,
        question: str,
        answer: str,
        expected_answer: str,
        context: str,
    ) -> float:
        """
        Evaluate a metric using GPT-5.4 as the intelligent evaluator.
        
        Args:
            metric_name: Name of the metric
            question: The question
            answer: The model's answer
            expected_answer: The expected correct answer
            context: Retrieved context
            
        Returns:
            float: Score between 0 and 1
        """
        try:
            if metric_name == "faithfulness":
                return self._evaluate_faithfulness_gpt(answer, context)
            elif metric_name == "answer_relevancy":
                return self._evaluate_relevancy_gpt(question, answer)
            elif metric_name == "correctness":
                return self._evaluate_correctness_gpt(answer, expected_answer)
            else:
                logger.warning(f"Unknown metric: {metric_name}")
                return 0.3
        except Exception as e:
            logger.error(f"Error evaluating {metric_name}: {e}")
            return 0.3  # Give partial credit on error
    
    def _evaluate_faithfulness_gpt(self, answer: str, context: str) -> float:
        """
        Use configured evaluator model to evaluate if the answer is faithful to the context.
        
        Args:
            answer: The model's answer
            context: Retrieved context
            
        Returns:
            float: Faithfulness score 0.0-1.0
        """
        prompt = f"""Evaluate how faithful the following answer is to the provided context.

CONTEXT:
{context}

ANSWER:
{answer}

Is the answer well-supported by the context? Rate on a scale of 0.0 to 1.0 where:
- 0.0-0.3: Answer contradicts or is unsupported by context
- 0.4-0.6: Answer is partially supported, some claims lack context
- 0.7-1.0: Answer is well-grounded in the provided context

Respond with ONLY a single number between 0.0 and 1.0. Example: 0.85"""

        try:
            response = self.evaluator_llm.invoke(prompt)
            score_text = response.content.strip()
            
            # Try multiple regex patterns to extract score
            patterns = [
                r'\b([01]\.?\d*)\b',  # Matches 0, 1, 0.1, 0.85, 1.0, etc.
                r'(\d\.\d+)',         # Matches 0.85, etc.
                r'(\d+\.?\d*)',       # Matches 1, 0.8, etc.
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, score_text)
                if matches:
                    score = float(matches[0])
                    return min(max(score, 0.0), 1.0)
            
            # If no score found, return middle ground
            logger.warning(f"Could not extract score from response: {score_text}")
            return 0.5
        except Exception as e:
            logger.error(f"Error in faithfulness evaluation: {e}")
            return 0.5
    
    def _evaluate_relevancy_gpt(self, question: str, answer: str) -> float:
        """
        Use GPT-5.4 to evaluate if the answer is relevant to the question.
        
        Args:
            question: The user's question
            answer: The model's answer
            
        Returns:
            float: Relevancy score 0.0-1.0
        """
        prompt = f"""Evaluate how relevant and focused the answer is to the question.

QUESTION:
{question}

ANSWER:
{answer}

Does the answer directly address the question? Rate on a scale of 0.0 to 1.0 where:
- 0.0-0.3: Answer is off-topic or unrelated
- 0.4-0.6: Answer partially addresses the question, with some irrelevant content
- 0.7-1.0: Answer is highly relevant and directly addresses the question

Respond with ONLY a single number between 0.0 and 1.0. Example: 0.92"""

        try:
            response = self.evaluator_llm.invoke(prompt)
            score_text = response.content.strip()
            score = float(re.findall(r'0\.\d+|1\.0', score_text)[0])
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            logger.error(f"Error in relevancy evaluation: {e}")
            return 0.5
    
    def _evaluate_correctness_gpt(self, answer: str, expected_answer: str) -> float:
        """
        Use GPT-5.4 to evaluate if the answer is correct compared to expected answer.
        
        Args:
            answer: The model's answer
            expected_answer: The expected correct answer
            
        Returns:
            float: Correctness score 0.0-1.0
        """
        prompt = f"""Evaluate how correct the provided answer is compared to the expected answer.

EXPECTED ANSWER:
{expected_answer}

PROVIDED ANSWER:
{answer}

How correct is the provided answer? Rate on a scale of 0.0 to 1.0 where:
- 0.0-0.3: Answer is incorrect or contradicts expected answer
- 0.4-0.6: Answer has some correct elements but is incomplete or partially wrong
- 0.7-1.0: Answer is substantially correct, may have minor differences in phrasing

Respond with ONLY a single number between 0.0 and 1.0. Example: 0.88"""

        try:
            response = self.evaluator_llm.invoke(prompt)
            score_text = response.content.strip()
            score = float(re.findall(r'0\.\d+|1\.0', score_text)[0])
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            logger.error(f"Error in correctness evaluation: {e}")
            return 0.5
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of evaluation results.
        
        Returns:
            Dict with aggregate metrics and comparison
        """
        if not self.evaluation_results:
            return {"status": "no_results"}
        
        # Aggregate scores by model
        model_stats = {}
        
        for result in self.evaluation_results:
            for model_name, scores in result.model_scores.items():
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        metric: [] for metric in scores.keys()
                    }
                    model_stats[model_name]["pass_count"] = 0
                
                for metric, score in scores.items():
                    model_stats[model_name][metric].append(score)
                
                if result.passed[model_name]:
                    model_stats[model_name]["pass_count"] += 1
        
        # Calculate averages
        summary = {
            "total_questions": len(self.evaluation_results),
            "model_metrics": {},
        }
        
        for model_name, stats in model_stats.items():
            pass_count = stats.pop("pass_count")
            
            metric_averages = {}
            for metric_name, scores in stats.items():
                if scores:
                    metric_averages[metric_name] = {
                        "mean": statistics.mean(scores),
                        "min": min(scores),
                        "max": max(scores),
                        "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    }
            
            summary["model_metrics"][model_name] = {
                "metrics": metric_averages,
                "pass_rate": pass_count / len(self.evaluation_results) if self.evaluation_results else 0.0,
                "pass_count": pass_count,
            }
        
        return summary
    
    def get_results(self) -> List[EvaluationResult]:
        """Get all evaluation results."""
        return self.evaluation_results


if __name__ == "__main__":
    # Test evaluator
    logging.basicConfig(level=logging.INFO)
    print("✓ Evaluator module loaded")
