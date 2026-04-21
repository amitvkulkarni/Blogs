"""
WARD TEST SUITE — ml_pipeline.py
==================================
Tests for data pipeline and ML model utilities.
Run with:  ward --path test_ward_ml.py
"""

from ward import test, fixture, each, raises
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


@test("validate_schema: passes when all required fields are present and non-null")
def _():
    record = {"sensor_id": "A1", "temperature": 22.5, "timestamp": "2026-01-01"}
    assert validate_schema(record, ["sensor_id", "temperature", "timestamp"]) is True


@test("validate_schema: fails when a required field is missing")
def _():
    record = {"sensor_id": "A1", "temperature": 22.5}
    assert validate_schema(record, ["sensor_id", "temperature", "timestamp"]) is False


@test("validate_schema: fails when a required field is None")
def _():
    record = {"sensor_id": "A1", "temperature": None}
    assert validate_schema(record, ["sensor_id", "temperature"]) is False


@test("remove_nulls: strips None values from a list")
def _():
    assert remove_nulls([1.0, None, 3.0, None, 5.0]) == [1.0, 3.0, 5.0]


@test("remove_nulls: returns empty list when all values are None")
def _():
    assert remove_nulls([None, None]) == []


@test("check_value_range: returns only the outlier values")
def _():
    values = [0.5, 1.2, 3.1, -0.1, 2.0]
    assert check_value_range(values, 0.0, 3.0) == [3.1, -0.1]


@test("check_value_range: returns empty list when all values are in range")
def _():
    assert check_value_range([1.0, 2.0, 3.0], 0.0, 5.0) == []


@test("deduplicate: removes duplicate records keeping first occurrence")
def _():
    records = [
        {"id": "m1", "accuracy": 0.91},
        {"id": "m2", "accuracy": 0.88},
        {"id": "m1", "accuracy": 0.93},
    ]
    result = deduplicate(records, key="id")
    assert len(result) == 2
    assert result[0]["accuracy"] == 0.91  # first occurrence kept


# ---------------------------------------------------------------------------
# 2. NORMALIZATION & STANDARDIZATION  (parametrized with each())
# ---------------------------------------------------------------------------


@test("normalize_minmax({values}) produces {expected}")
def _(
    values=each([0.0, 5.0, 10.0], [10.0, 20.0, 30.0], [-2.0, 0.0, 2.0]),
    expected=each([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]),
):
    result = normalize_minmax(values)
    assert [round(r, 4) for r in result] == expected


@test("normalize_minmax raises ValueError on empty list")
def _():
    with raises(ValueError):
        normalize_minmax([])


@test("normalize_minmax raises ValueError when all values are identical")
def _():
    with raises(ValueError):
        normalize_minmax([5.0, 5.0, 5.0])


@test("standardize_zscore: mean of standardized values is effectively zero")
def _():
    result = standardize_zscore([2.0, 4.0, 6.0])
    mean = sum(result) / len(result)
    assert abs(mean) < 1e-9


@test("standardize_zscore raises ValueError on empty list")
def _():
    with raises(ValueError):
        standardize_zscore([])


@test("standardize_zscore raises ValueError on zero standard deviation")
def _():
    with raises(ValueError):
        standardize_zscore([3.0, 3.0, 3.0])


# ---------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------


@test("train_test_split: produces correctly sized train and test sets")
def _():
    train, test_set = train_test_split(list(range(100)), test_ratio=0.2, seed=42)
    assert len(train) == 80
    assert len(test_set) == 20


@test("train_test_split: train and test sets have no overlap")
def _():
    data = list(range(50))
    train, test_set = train_test_split(data, test_ratio=0.3, seed=0)
    assert set(train).isdisjoint(set(test_set))


@test("train_test_split: union of train and test covers entire dataset")
def _():
    data = list(range(50))
    train, test_set = train_test_split(data, test_ratio=0.3, seed=0)
    assert sorted(train + test_set) == data


@test("train_test_split raises ValueError for out-of-range test_ratio")
def _():
    with raises(ValueError):
        train_test_split([1, 2, 3], test_ratio=1.5)


# ---------------------------------------------------------------------------
# 4. CLASSIFICATION METRICS  (parametrized with each())
# ---------------------------------------------------------------------------

Y_TRUE = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
Y_PRED = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]


@test("accuracy: perfect predictions return 1.0")
def _():
    assert accuracy([1, 0, 1], [1, 0, 1]) == 1.0


@test("accuracy: all wrong predictions return 0.0")
def _():
    assert accuracy([1, 1, 1], [0, 0, 0]) == 0.0


@test("accuracy([{y_true}], [{y_pred}]) = {expected_acc}")
def _(
    y_true=each([1, 0, 1, 1], [1, 1, 0, 0]),
    y_pred=each([1, 0, 0, 1], [1, 0, 1, 0]),
    expected_acc=each(0.75, 0.50),
):
    assert accuracy(y_true, y_pred) == expected_acc


@test("precision on sample data equals 0.8 (4 TP, 1 FP)")
def _():
    assert round(precision(Y_TRUE, Y_PRED), 2) == 0.8


@test("recall on sample data equals 0.8 (4 TP, 1 FN)")
def _():
    assert round(recall(Y_TRUE, Y_PRED), 2) == 0.8


@test("f1_score on sample data equals 0.8")
def _():
    assert round(f1_score(Y_TRUE, Y_PRED), 2) == 0.8


@test("accuracy raises ValueError when y_true and y_pred lengths differ")
def _():
    with raises(ValueError):
        accuracy([1, 0], [1, 0, 1])


# ---------------------------------------------------------------------------
# 5. REGRESSION METRICS
# ---------------------------------------------------------------------------


@test("mean_absolute_error: basic calculation returns 1.0")
def _():
    assert mean_absolute_error([3.0, 5.0], [2.0, 4.0]) == 1.0


@test("root_mean_squared_error: basic calculation returns 1.0")
def _():
    result = root_mean_squared_error([3.0, 5.0], [2.0, 4.0])
    assert round(result, 4) == 1.0


@test("mean_absolute_error: perfect predictions return 0.0")
def _():
    assert mean_absolute_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


@test("mean_absolute_error raises ValueError on empty lists")
def _():
    with raises(ValueError):
        mean_absolute_error([], [])


# ---------------------------------------------------------------------------
# 6. DATAPIPELINE  (fixtures + chained transforms)
# ---------------------------------------------------------------------------


@fixture
def empty_pipeline():
    yield DataPipeline()


@fixture
def cleaning_pipeline():
    pipe = DataPipeline()
    pipe.add_step("remove_nulls", remove_nulls)
    pipe.add_step("normalize", normalize_minmax)
    yield pipe


@test("DataPipeline: runs remove_nulls then normalize in order")
def _(pipe=cleaning_pipeline):
    result = pipe.run([10.0, None, 20.0, 30.0])
    assert result == [0.0, 0.5, 1.0]


@test("DataPipeline: step names are recorded in insertion order")
def _(pipe=cleaning_pipeline):
    assert pipe.step_names == ["remove_nulls", "normalize"]


@test("DataPipeline: raises RuntimeError when run with no steps")
def _(pipe=empty_pipeline):
    with raises(RuntimeError):
        pipe.run([1.0, 2.0])


@test("DataPipeline: raises TypeError when a non-callable is added as a step")
def _(pipe=empty_pipeline):
    with raises(TypeError):
        pipe.add_step("bad_step", "not_a_function")


@test("DataPipeline: clear() removes all steps")
def _(pipe=cleaning_pipeline):
    pipe.clear()
    assert pipe.step_names == []


# ---------------------------------------------------------------------------
# 7. MODELREGISTRY
# ---------------------------------------------------------------------------


@fixture
def registry():
    reg = ModelRegistry()
    reg.register("churn_model", "v1", {"accuracy": 0.82, "f1": 0.79})
    reg.register("churn_model", "v2", {"accuracy": 0.87, "f1": 0.85})
    reg.register("churn_model", "v3", {"accuracy": 0.84, "f1": 0.81})
    yield reg


@test("ModelRegistry: stores and retrieves metrics for a registered version")
def _(reg=registry):
    assert reg.get_metrics("churn_model", "v1")["accuracy"] == 0.82


@test("ModelRegistry: lists all registered versions for a model")
def _(reg=registry):
    assert reg.list_versions("churn_model") == ["v1", "v2", "v3"]


@test("ModelRegistry: promote sets the production version correctly")
def _(reg=registry):
    reg.promote("churn_model", "v2")
    assert reg.get_production_version("churn_model") == "v2"


@test("ModelRegistry: best_version returns highest accuracy version")
def _(reg=registry):
    assert reg.best_version("churn_model", "accuracy") == "v2"


@test("ModelRegistry: best_version works with lower_is_better for RMSE")
def _():
    reg = ModelRegistry()
    reg.register("loss_model", "v1", {"rmse": 0.45})
    reg.register("loss_model", "v2", {"rmse": 0.31})
    assert reg.best_version("loss_model", "rmse", higher_is_better=False) == "v2"


@test("ModelRegistry: raises KeyError for unregistered model")
def _(reg=registry):
    with raises(KeyError):
        reg.get_metrics("unknown_model", "v1")


@test("ModelRegistry: raises KeyError when no production version is set")
def _(reg=registry):
    with raises(KeyError):
        reg.get_production_version("churn_model")


@test("ModelRegistry: returns empty list for unknown model versions")
def _(reg=registry):
    assert reg.list_versions("nonexistent_model") == []


# ---------------------------------------------------------------------------
# 8. INTENTIONAL FAILURE  (demonstrates failure output)
# ---------------------------------------------------------------------------


@test("[INTENTIONAL FAIL] perfect accuracy should be 1.0 not 0.5 (expect failure)")
def _():
    assert (
        accuracy([1, 0, 1], [1, 0, 1]) == 0.5
    ), "Demo: perfect accuracy should be 1.0 not 0.5"
