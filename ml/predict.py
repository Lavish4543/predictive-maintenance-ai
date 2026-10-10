
import os
import joblib
import numpy as np
import pandas as pd


# Find the saved model relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "predictive_maintenance_model.joblib"
)

# Load the model once when this module is imported
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
MODEL_NAME = bundle["model_name"]
THRESHOLD = float(bundle["threshold"])
FEATURE_NAMES = bundle["feature_names"]


def predict_failure(machine_data):
    """
    Predict machine failure risk from sensor readings.

    Required input:
        Air temperature [K]
        Process temperature [K]
        Rotational speed [rpm]
        Torque [Nm]
        Tool wear [min]
        Type: L, M, or H
    """

    required_fields = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type"
    ]

    # Validate required fields
    missing_fields = [
        field for field in required_fields
        if field not in machine_data
    ]

    if missing_fields:
        raise ValueError(
            f"Missing required fields: {missing_fields}"
        )

    # Validate machine type
    machine_type = str(machine_data["Type"]).upper()

    if machine_type not in ["L", "M", "H"]:
        raise ValueError(
            "Type must be L, M, or H"
        )

    # Convert sensor values to numeric values
    sensor_fields = required_fields[:-1]

    try:
        sensor_values = {
            field: float(machine_data[field])
            for field in sensor_fields
        }
    except (ValueError, TypeError):
        raise ValueError(
            "All sensor readings must be numeric"
        )

    # Reject NaN and infinite values
    if not all(
        np.isfinite(value)
        for value in sensor_values.values()
    ):
        raise ValueError(
            "Sensor readings must be finite numbers"
        )

    # Basic physical input validation
    if sensor_values["Air temperature [K]"] <= 0:
        raise ValueError("Air temperature must be positive")

    if sensor_values["Process temperature [K]"] <= 0:
        raise ValueError("Process temperature must be positive")

    if sensor_values["Rotational speed [rpm]"] <= 0:
        raise ValueError("Rotational speed must be positive")

    if sensor_values["Torque [Nm]"] < 0:
        raise ValueError("Torque cannot be negative")

    if sensor_values["Tool wear [min]"] < 0:
        raise ValueError("Tool wear cannot be negative")

    # Build the feature row expected by the trained model
    row = {
        **sensor_values,
        "Type_H": int(machine_type == "H"),
        "Type_L": int(machine_type == "L"),
        "Type_M": int(machine_type == "M")
    }

    input_df = pd.DataFrame([row])

    # Preserve exact feature names and order
    input_df = input_df.reindex(
        columns=FEATURE_NAMES,
        fill_value=0
    )

    # Predict failure probability and binary alert
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
        )
    }
