"""
ml_pipeline.py
==============
Code-under-test for the pytest vs ward framework comparison,
scoped to professionals working with data pipelines and ML models.

Covers:
  - Data validation and cleaning
  - Numeric normalization and standardization
  - Train / test splitting
  - Classification and regression metrics
  - A chainable DataPipeline class
  - A ModelRegistry for tracking model versions
"""

import math
import random
from typing import Any


# ---------------------------------------------------------------------------
# Data validation & cleaning
# ---------------------------------------------------------------------------


def validate_schema(record: dict, required_fields: list[str]) -> bool:
    """Return True if all required_fields are present and non-None in record."""
    return all(
        field in record and record[field] is not None for field in required_fields
    )


def remove_nulls(values: list) -> list:
    """Strip None values from a list."""
    return [v for v in values if v is not None]


def check_value_range(values: list[float], low: float, high: float) -> list[float]:
    """Return values that fall outside [low, high]."""
    return [v for v in values if not (low <= v <= high)]


def deduplicate(records: list[dict], key: str) -> list[dict]:
    """Remove duplicate dicts keeping first occurrence, based on a key field."""
    seen: set = set()
    result = []
    for rec in records:
        k = rec[key]
        if k not in seen:
            seen.add(k)
            result.append(rec)
    return result


# ---------------------------------------------------------------------------
# Normalization & standardization
# ---------------------------------------------------------------------------


def normalize_minmax(values: list[float]) -> list[float]:
    """Min-max scale values to [0.0, 1.0]."""
    if not values:
        raise ValueError("Cannot normalize an empty list")
    lo, hi = min(values), max(values)
    if lo == hi:
        raise ValueError("Cannot normalize: all values are identical")
    return [(v - lo) / (hi - lo) for v in values]


def standardize_zscore(values: list[float]) -> list[float]:
    """Z-score standardize: (x - mean) / std."""
    if not values:
        raise ValueError("Cannot standardize an empty list")
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    if std == 0:
        raise ValueError("Cannot standardize: zero standard deviation")
    return [(v - mean) / std for v in values]


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------


def train_test_split(
    data: list,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list, list]:
    """Randomly split data into (train, test) sets."""
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1 (exclusive)")
    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)
    split = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split], shuffled[split:]


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def accuracy(y_true: list, y_pred: list) -> float:
    """Fraction of correctly predicted labels."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("Cannot compute accuracy on empty lists")
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def precision(y_true: list, y_pred: list, positive_label: Any = 1) -> float:
    """Precision = TP / (TP + FP)."""
    tp = sum(
        t == positive_label and p == positive_label for t, p in zip(y_true, y_pred)
    )
    fp = sum(
        t != positive_label and p == positive_label for t, p in zip(y_true, y_pred)
    )
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall(y_true: list, y_pred: list, positive_label: Any = 1) -> float:
    """Recall = TP / (TP + FN)."""
    tp = sum(
        t == positive_label and p == positive_label for t, p in zip(y_true, y_pred)
    )
    fn = sum(
        t == positive_label and p != positive_label for t, p in zip(y_true, y_pred)
    )
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(y_true: list, y_pred: list, positive_label: Any = 1) -> float:
    """Harmonic mean of precision and recall."""
    p = precision(y_true, y_pred, positive_label)
    r = recall(y_true, y_pred, positive_label)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------


def mean_absolute_error(y_true: list[float], y_pred: list[float]) -> float:
    """Average absolute difference between predictions and actuals."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("Cannot compute MAE on empty lists")
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def root_mean_squared_error(y_true: list[float], y_pred: list[float]) -> float:
    """Square root of the average squared differences."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("Cannot compute RMSE on empty lists")
    mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)
    return math.sqrt(mse)


# ---------------------------------------------------------------------------
# DataPipeline — chainable transformation steps
# ---------------------------------------------------------------------------


class DataPipeline:
    """
    Chain named transformation functions and apply them sequentially
    to a dataset.

    Example
    -------
    pipe = DataPipeline()
    pipe.add_step("remove_nulls", remove_nulls)
    pipe.add_step("normalize", normalize_minmax)
    result = pipe.run([1.0, None, 3.0, 5.0])
    """

    def __init__(self):
        self._steps: list[tuple[str, callable]] = []

    def add_step(self, name: str, func: callable) -> "DataPipeline":
        """Append a named step. Returns self for chaining."""
        if not callable(func):
            raise TypeError(f"Step '{name}' must be callable")
        self._steps.append((name, func))
        return self

    @property
    def step_names(self) -> list[str]:
        return [name for name, _ in self._steps]

    def run(self, data: list) -> list:
        """Execute all steps in order and return the transformed data."""
        if not self._steps:
            raise RuntimeError(
                "Pipeline has no steps — add at least one step before running"
            )
        result = data
        for _, func in self._steps:
            result = func(result)
        return result

    def clear(self) -> None:
        self._steps.clear()


# ---------------------------------------------------------------------------
# ModelRegistry — lightweight model version tracker
# ---------------------------------------------------------------------------


class ModelRegistry:
    """
    In-memory registry for tracking ML model versions and their metrics.

    Example
    -------
    registry = ModelRegistry()
    registry.register("churn_model", "v1", {"accuracy": 0.87, "f1": 0.83})
    registry.promote("churn_model", "v1")
    """

    def __init__(self):
        self._models: dict[str, dict[str, dict]] = {}
        self._production: dict[str, str] = {}

    def register(self, model_name: str, version: str, metrics: dict) -> None:
        """Register a model version with its evaluation metrics."""
        if model_name not in self._models:
            self._models[model_name] = {}
        self._models[model_name][version] = {"metrics": metrics, "promoted": False}

    def get_metrics(self, model_name: str, version: str) -> dict:
        """Retrieve metrics for a registered model version."""
        if model_name not in self._models:
            raise KeyError(f"Model '{model_name}' not found in registry")
        if version not in self._models[model_name]:
            raise KeyError(f"Version '{version}' of model '{model_name}' not found")
        return self._models[model_name][version]["metrics"]

    def promote(self, model_name: str, version: str) -> None:
        """Promote a version to production."""
        if model_name not in self._models or version not in self._models[model_name]:
            raise KeyError(
                f"Cannot promote: '{model_name}' version '{version}' not registered"
            )
        self._production[model_name] = version

    def get_production_version(self, model_name: str) -> str:
        """Return the currently promoted production version."""
        if model_name not in self._production:
            raise KeyError(f"No production version set for '{model_name}'")
        return self._production[model_name]

    def list_versions(self, model_name: str) -> list[str]:
        """List all registered versions for a model."""
        if model_name not in self._models:
            return []
        return list(self._models[model_name].keys())

    def best_version(
        self, model_name: str, metric: str, higher_is_better: bool = True
    ) -> str:
        """Return the version with the best value for a given metric."""
        if model_name not in self._models:
            raise KeyError(f"Model '{model_name}' not found")
        versions = self._models[model_name]
        return max(
            versions,
            key=lambda v: (
                versions[v]["metrics"].get(metric, float("-inf"))
                if higher_is_better
                else -versions[v]["metrics"].get(metric, float("inf"))
            ),
        )
