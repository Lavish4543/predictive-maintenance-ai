
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "predictive_maintenance_model_candidate.joblib"
)

DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"


def main():
    # Load the candidate bundle.
    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    threshold = float(bundle["threshold"])
    feature_names = bundle["feature_names"]

    print("MODEL BUNDLE CHECK")
    print("------------------")
    print("Model:", bundle["model_name"])
    print("Threshold:", threshold)
    print("Training rows:", bundle["training_rows"])
    print("Features:", feature_names)

    # Check that the saved model has the expected features.
    if list(model.feature_names_in_) != list(feature_names):
        raise ValueError(
            "Saved model features do not match bundle feature names."
        )

    # Use an actual dataset row for an integration smoke test.
    df = pd.read_csv(DATA_PATH)

    raw_row = df.iloc[[0]].copy()
    actual_label = int(raw_row["Machine failure"].iloc[0])

    drop_columns = [
        "UDI",
        "Product ID",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    features = raw_row.drop(columns=drop_columns)

    features = pd.get_dummies(
        features,
        columns=["Type"],
        dtype=int,
    )

    # Enforce the exact feature order used in training.
    features = features.reindex(
        columns=feature_names,
        fill_value=0,
    )

    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Test input contains non-finite values.")

    probability = float(
        model.predict_proba(features)[0, 1]
    )

    prediction = int(probability >= threshold)

    print("\nPREDICTION SMOKE TEST")
    print("---------------------")
    print("Actual dataset label:", actual_label)
    print("Failure probability:", round(probability, 4))
    print("Threshold:", threshold)
    print("Predicted failure:", prediction)
    print(
        "Status:",
        "FAILURE_ALERT" if prediction == 1
        else "NO_FAILURE_ALERT",
    )

    # Basic output sanity checks.
    assert 0.0 <= probability <= 1.0
    assert prediction in (0, 1)
    assert threshold > 0.0 and threshold < 1.0

    print("\nPASS: Candidate bundle smoke test succeeded.")


if __name__ == "__main__":
    main()
