
import os
import joblib
import numpy as np
import pandas as pd


# Load the candidate model relative to this file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "predictive_maintenance_model_candidate.joblib",
)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

bundle = joblib.load(MODEL_PATH)

required_bundle_keys = {
    "model",
    "model_name",
    "threshold",
    "feature_names",
}

if not isinstance(bundle, dict):
    raise ValueError("Model bundle must be a dictionary.")

missing_keys = required_bundle_keys - set(bundle.keys())

if missing_keys:
    raise ValueError(
        f"Model bundle is missing keys: {sorted(missing_keys)}"
    )

model = bundle["model"]
MODEL_NAME = bundle["model_name"]
THRESHOLD = float(bundle["threshold"])
FEATURE_NAMES = list(bundle["feature_names"])

if not 0.0 < THRESHOLD < 1.0:
    raise ValueError(
        "Model threshold must be between 0 and 1."
    )


SENSOR_FIELDS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

REQUIRED_FIELDS = SENSOR_FIELDS + ["Type"]


def predict_failure(machine_data: dict) -> dict:
    """
    Predict machine failure risk from sensor readings.

    Required fields:
        Air temperature [K]
        Process temperature [K]
        Rotational speed [rpm]
        Torque [Nm]
        Tool wear [min]
        Type: L, M, or H
    """

    # Validate the overall request structure.
    if not isinstance(machine_data, dict):
        raise ValueError(
            "Input must be a JSON object / Python dictionary."
        )

    # Check required fields first.
    missing_fields = [
        field for field in REQUIRED_FIELDS
        if field not in machine_data
    ]

    if missing_fields:
        raise ValueError(
            f"Missing required fields: {missing_fields}"
        )

    # Validate machine type.
    if not isinstance(machine_data["Type"], str):
        raise ValueError("Type must be L, M, or H.")

    machine_type = machine_data["Type"].upper()

    if machine_type not in ["L", "M", "H"]:
        raise ValueError("Type must be L, M, or H.")

    # Reject boolean sensor values explicitly.
    for field in SENSOR_FIELDS:
        value = machine_data[field]

        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                f"{field} must be numeric, not boolean."
            )

    # Convert sensor readings to numeric values.
    try:
        sensor_values = {
            field: float(machine_data[field])
            for field in SENSOR_FIELDS
        }
    except (ValueError, TypeError, OverflowError):
        raise ValueError(
            "All sensor readings must be numeric."
        ) from None

    # Reject NaN and infinite values.
    if not all(
        np.isfinite(value)
        for value in sensor_values.values()
    ):
        raise ValueError(
            "Sensor readings must be finite numbers."
        )

    # Basic physical input validation.
    if sensor_values["Air temperature [K]"] <= 0:
        raise ValueError("Air temperature must be positive.")

    if sensor_values["Process temperature [K]"] <= 0:
        raise ValueError("Process temperature must be positive.")

    if sensor_values["Rotational speed [rpm]"] <= 0:
        raise ValueError("Rotational speed must be positive.")

    if sensor_values["Torque [Nm]"] < 0:
        raise ValueError("Torque cannot be negative.")

    if sensor_values["Tool wear [min]"] < 0:
        raise ValueError("Tool wear cannot be negative.")

    # Build the feature row expected by the trained model.
    row = {
        **sensor_values,
        "Type_H": int(machine_type == "H"),
        "Type_L": int(machine_type == "L"),
        "Type_M": int(machine_type == "M"),
    }

    input_df = pd.DataFrame([row])

    # Preserve the exact feature names and order.
    input_df = input_df.reindex(
        columns=FEATURE_NAMES,
        fill_value=0,
    )

    # Predict probability and threshold-based alert.
    probability = float(
        model.predict_proba(input_df)[0, 1]
    )

    predicted_failure = int(
        probability >= THRESHOLD
    )

    return {
        "model_name": MODEL_NAME,
        "failure_probability": probability,
        "threshold": THRESHOLD,
        "predicted_failure": predicted_failure,
        "status": (
            "FAILURE_ALERT"
            if predicted_failure == 1
            else "NO_FAILURE_ALERT"
        ),
    }
