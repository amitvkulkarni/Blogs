"""
PYTEST TEST SUITE — ml_pipeline.py
====================================
Tests for data pipeline and ML model utilities.
Run with:  pytest test_pytest_ml.py -v
"""

import pytest
from ml_pipeline import (
    validate_schema,
    remove_nulls,
    check_value_range,
    deduplicate,
    normalize_minmax,
    standardize_zscore,
    train_test_split,
    accuracy,
    precision,
    recall,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
    DataPipeline,
    ModelRegistry,
)


# ---------------------------------------------------------------------------
# 1. DATA VALIDATION & CLEANING
# ---------------------------------------------------------------------------


def test_validate_schema_passes_when_all_fields_present():
    record = {"sensor_id": "A1", "temperature": 22.5, "timestamp": "2026-01-01"}
    assert validate_schema(record, ["sensor_id", "temperature", "timestamp"]) is True


def test_validate_schema_fails_when_field_missing():
    record = {"sensor_id": "A1", "temperature": 22.5}
    assert validate_schema(record, ["sensor_id", "temperature", "timestamp"]) is False


def test_validate_schema_fails_when_field_is_none():
    record = {"sensor_id": "A1", "temperature": None}
    assert validate_schema(record, ["sensor_id", "temperature"]) is False


def test_remove_nulls_strips_none_values():
    assert remove_nulls([1.0, None, 3.0, None, 5.0]) == [1.0, 3.0, 5.0]


def test_remove_nulls_returns_empty_for_all_none():
    assert remove_nulls([None, None]) == []


def test_check_value_range_returns_outliers():
    values = [0.5, 1.2, 3.1, -0.1, 2.0]
    assert check_value_range(values, 0.0, 3.0) == [3.1, -0.1]


def test_check_value_range_returns_empty_when_all_in_range():
    assert check_value_range([1.0, 2.0, 3.0], 0.0, 5.0) == []


def test_deduplicate_removes_duplicate_records():
    records = [
        {"id": "m1", "accuracy": 0.91},
        {"id": "m2", "accuracy": 0.88},
        {"id": "m1", "accuracy": 0.93},
    ]
    result = deduplicate(records, key="id")
    assert len(result) == 2
    assert result[0]["id"] == "m1"
    assert result[0]["accuracy"] == 0.91  # keeps first


# ---------------------------------------------------------------------------
# 2. NORMALIZATION & STANDARDIZATION  (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0.0, 5.0, 10.0], [0.0, 0.5, 1.0]),
        ([10.0, 20.0, 30.0], [0.0, 0.5, 1.0]),
        ([-2.0, 0.0, 2.0], [0.0, 0.5, 1.0]),
    ],
)
def test_normalize_minmax(values, expected):
    result = normalize_minmax(values)
    assert [round(r, 4) for r in result] == expected


def test_normalize_minmax_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        normalize_minmax([])


def test_normalize_minmax_raises_when_all_identical():
    with pytest.raises(ValueError, match="identical"):
        normalize_minmax([5.0, 5.0, 5.0])


def test_standardize_zscore_mean_is_zero():
    result = standardize_zscore([2.0, 4.0, 6.0])
    mean = sum(result) / len(result)
    assert abs(mean) < 1e-9


def test_standardize_zscore_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        standardize_zscore([])


def test_standardize_zscore_raises_on_zero_std():
    with pytest.raises(ValueError, match="zero standard deviation"):
        standardize_zscore([3.0, 3.0, 3.0])


# ---------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------


def test_train_test_split_correct_sizes():
    data = list(range(100))
    train, test = train_test_split(data, test_ratio=0.2, seed=42)
    assert len(train) == 80
    assert len(test) == 20


def test_train_test_split_no_overlap():
    data = list(range(50))
    train, test = train_test_split(data, test_ratio=0.3, seed=0)
    assert set(train).isdisjoint(set(test))


def test_train_test_split_covers_all_data():
    data = list(range(50))
    train, test = train_test_split(data, test_ratio=0.3, seed=0)
    assert sorted(train + test) == data


def test_train_test_split_invalid_ratio_raises():
    with pytest.raises(ValueError):
        train_test_split([1, 2, 3], test_ratio=1.5)


# ---------------------------------------------------------------------------
# 4. CLASSIFICATION METRICS  (parametrized)
# ---------------------------------------------------------------------------

Y_TRUE = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
Y_PRED = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]


def test_accuracy_perfect():
    assert accuracy([1, 0, 1], [1, 0, 1]) == 1.0


def test_accuracy_zero():
    assert accuracy([1, 1, 1], [0, 0, 0]) == 0.0


@pytest.mark.parametrize(
    "y_true, y_pred, expected_acc",
    [
        ([1, 0, 1, 1], [1, 0, 0, 1], 0.75),
        ([1, 1, 0, 0], [1, 0, 1, 0], 0.50),
    ],
)
def test_accuracy_parametrized(y_true, y_pred, expected_acc):
    assert accuracy(y_true, y_pred) == expected_acc


def test_precision_value():
    # TP=4, FP=1 → precision = 4/5 = 0.8
    assert round(precision(Y_TRUE, Y_PRED), 2) == 0.8


def test_recall_value():
    # TP=4, FN=1 → recall = 4/5 = 0.8
    assert round(recall(Y_TRUE, Y_PRED), 2) == 0.8


def test_f1_value():
    assert round(f1_score(Y_TRUE, Y_PRED), 2) == 0.8


def test_accuracy_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        accuracy([1, 0], [1, 0, 1])


# ---------------------------------------------------------------------------
# 5. REGRESSION METRICS
# ---------------------------------------------------------------------------


def test_mae_basic():
    assert mean_absolute_error([3.0, 5.0], [2.0, 4.0]) == 1.0


def test_rmse_basic():
    result = root_mean_squared_error([3.0, 5.0], [2.0, 4.0])
    assert round(result, 4) == 1.0


def test_mae_perfect_predictions():
    assert mean_absolute_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_mae_raises_on_empty():
    with pytest.raises(ValueError):
        mean_absolute_error([], [])


# ---------------------------------------------------------------------------
# 6. DATAPIPELINE  (fixtures + chained transforms)
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_pipeline():
    return DataPipeline()


@pytest.fixture
def cleaning_pipeline():
    pipe = DataPipeline()
    pipe.add_step("remove_nulls", remove_nulls)
    pipe.add_step("normalize", normalize_minmax)
    return pipe


def test_pipeline_runs_steps_in_order(cleaning_pipeline):
    result = cleaning_pipeline.run([10.0, None, 20.0, 30.0])
    assert result == [0.0, 0.5, 1.0]


def test_pipeline_step_names_recorded(cleaning_pipeline):
    assert cleaning_pipeline.step_names == ["remove_nulls", "normalize"]


def test_pipeline_raises_when_no_steps(empty_pipeline):
    with pytest.raises(RuntimeError, match="no steps"):
        empty_pipeline.run([1.0, 2.0])


def test_pipeline_raises_on_non_callable(empty_pipeline):
    with pytest.raises(TypeError):
        empty_pipeline.add_step("bad_step", "not_a_function")


def test_pipeline_clear_removes_all_steps(cleaning_pipeline):
    cleaning_pipeline.clear()
    assert cleaning_pipeline.step_names == []


# ---------------------------------------------------------------------------
# 7. MODELREGISTRY
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    reg = ModelRegistry()
    reg.register("churn_model", "v1", {"accuracy": 0.82, "f1": 0.79})
    reg.register("churn_model", "v2", {"accuracy": 0.87, "f1": 0.85})
    reg.register("churn_model", "v3", {"accuracy": 0.84, "f1": 0.81})
    return reg


def test_registry_stores_metrics(registry):
    metrics = registry.get_metrics("churn_model", "v1")
    assert metrics["accuracy"] == 0.82


def test_registry_lists_all_versions(registry):
    assert registry.list_versions("churn_model") == ["v1", "v2", "v3"]


def test_registry_promote_sets_production(registry):
    registry.promote("churn_model", "v2")
    assert registry.get_production_version("churn_model") == "v2"


def test_registry_best_version_by_accuracy(registry):
    assert registry.best_version("churn_model", "accuracy") == "v2"


def test_registry_best_version_lower_is_better(registry):
    reg = ModelRegistry()
    reg.register("loss_model", "v1", {"rmse": 0.45})
    reg.register("loss_model", "v2", {"rmse": 0.31})
    assert reg.best_version("loss_model", "rmse", higher_is_better=False) == "v2"


def test_registry_raises_for_unknown_model(registry):
    with pytest.raises(KeyError):
        registry.get_metrics("unknown_model", "v1")


def test_registry_raises_when_no_production_set(registry):
    with pytest.raises(KeyError):
        registry.get_production_version("churn_model")


def test_registry_returns_empty_list_for_unknown_model(registry):
    assert registry.list_versions("nonexistent_model") == []


# ---------------------------------------------------------------------------
# 8. INTENTIONAL FAILURE  (demonstrates failure output)
# ---------------------------------------------------------------------------


def test_intentional_failure():
    """Intentionally wrong to demo failure output."""
    assert (
        accuracy([1, 0, 1], [1, 0, 1]) == 0.5
    ), "Demo: perfect accuracy should be 1.0 not 0.5"
