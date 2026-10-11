
import numpy as np
import pandas as pd
import pytest

from predict import predict_failure


# Use a real row from the project's dataset.
df = pd.read_csv("data/ai4i2020.csv")
row = df.iloc[0]

VALID_INPUT = {
    "Air temperature [K]": float(row["Air temperature [K]"]),
    "Process temperature [K]": float(row["Process temperature [K]"]),
    "Rotational speed [rpm]": float(row["Rotational speed [rpm]"]),
    "Torque [Nm]": float(row["Torque [Nm]"]),
    "Tool wear [min]": float(row["Tool wear [min]"]),
    "Type": str(row["Type"]),
}


def test_valid_input_returns_expected_fields():
    result = predict_failure(VALID_INPUT)

    expected_keys = {
        "model_name",
        "failure_probability",
        "threshold",
        "predicted_failure",
        "status",
    }

    assert set(result.keys()) == expected_keys


def test_probability_is_within_valid_range():
    result = predict_failure(VALID_INPUT)

    assert 0.0 <= result["failure_probability"] <= 1.0


def test_prediction_matches_threshold():
    result = predict_failure(VALID_INPUT)

    expected_prediction = int(
        result["failure_probability"] >= result["threshold"]
    )

    assert result["predicted_failure"] == expected_prediction


def test_status_matches_prediction():
    result = predict_failure(VALID_INPUT)

    expected_status = (
        "FAILURE_ALERT"
        if result["predicted_failure"] == 1
        else "NO_FAILURE_ALERT"
    )

    assert result["status"] == expected_status


def test_missing_sensor_field_is_rejected():
    data = VALID_INPUT.copy()
    del data["Torque [Nm]"]

    with pytest.raises(ValueError, match="Missing required fields"):
        predict_failure(data)


def test_invalid_machine_type_is_rejected():
    data = VALID_INPUT.copy()
    data["Type"] = "X"

    with pytest.raises(ValueError, match="Type must be"):
        predict_failure(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("Air temperature [K]", "abc"),
        ("Process temperature [K]", "unknown"),
        ("Rotational speed [rpm]", None),
    ],
)
def test_non_numeric_sensor_values_are_rejected(field, value):
    data = VALID_INPUT.copy()
    data[field] = value

    with pytest.raises(ValueError, match="numeric"):
        predict_failure(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("Air temperature [K]", np.nan),
        ("Process temperature [K]", np.inf),
        ("Rotational speed [rpm]", -np.inf),
    ],
)
def test_non_finite_sensor_values_are_rejected(field, value):
    data = VALID_INPUT.copy()
    data[field] = value

    with pytest.raises(ValueError, match="finite"):
        predict_failure(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("Air temperature [K]", 0),
        ("Process temperature [K]", 0),
        ("Rotational speed [rpm]", 0),
        ("Torque [Nm]", -1),
        ("Tool wear [min]", -1),
    ],
)
def test_invalid_physical_values_are_rejected(field, value):
    data = VALID_INPUT.copy()
    data[field] = value

    with pytest.raises(ValueError):
        predict_failure(data)


def test_same_input_gives_repeatable_result():
    first = predict_failure(VALID_INPUT)
    second = predict_failure(VALID_INPUT)

    assert first == second
