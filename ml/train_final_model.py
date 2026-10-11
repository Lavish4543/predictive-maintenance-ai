
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
MODEL_DIR = BASE_DIR / "models"

MIN_RECALL = 0.80
RANDOM_STATE = 42
N_SPLITS = 5

MODEL_PATH = (
    MODEL_DIR / "predictive_maintenance_model_candidate.joblib"
)

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


# --------------------------------------------------
# 2. Load and prepare dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

required_columns = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset is missing columns: {missing_columns}"
    )

df = df.drop(
    columns=[
        "UDI",
        "Product ID",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]
)

df = pd.get_dummies(
    df,
    columns=["Type"],
    dtype=int,
)

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"].astype(int)

if y.nunique() != 2:
    raise ValueError(
        "Machine failure must contain both classes 0 and 1."
    )

print("Dataset rows:", len(X))
print("Failure cases:", int(y.sum()))
print("Normal cases:", int((y == 0).sum()))
print("Features:", list(X.columns))


# --------------------------------------------------
# 3. Define the final model configuration
# --------------------------------------------------

def create_model():
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# --------------------------------------------------
# 4. Generate out-of-fold probabilities
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

print("\nGenerating out-of-fold probabilities...")

oof_probabilities = cross_val_predict(
    create_model(),
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=1,
)[:, 1]


# --------------------------------------------------
# 5. Select threshold using the recall requirement
# --------------------------------------------------

eligible_thresholds = []

for threshold in THRESHOLDS:
    predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    if recall >= MIN_RECALL:
        eligible_thresholds.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

if not eligible_thresholds:
    raise RuntimeError(
        "No threshold achieved the 80% recall target. "
        "No candidate model was saved."
    )

selected = max(
    eligible_thresholds,
    key=lambda item: (
        item["precision"],
        item["f1"],
        item["threshold"],
    ),
)

threshold = selected["threshold"]

oof_predictions = (
    oof_probabilities >= threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y,
    oof_predictions,
    labels=[0, 1],
).ravel()

print("\nTHRESHOLD SELECTION REPORT")
print("--------------------------")
print("Selection method: OOF predictions")
print("Minimum recall target:", MIN_RECALL)
print("Selected threshold:", threshold)
print("OOF precision:", round(
    precision_score(y, oof_predictions, zero_division=0), 4
))
print("OOF recall:", round(
    recall_score(y, oof_predictions, zero_division=0), 4
))
print("OOF F1-score:", round(
    f1_score(y, oof_predictions, zero_division=0), 4
))
print("OOF Average Precision:", round(
    average_precision_score(y, oof_probabilities), 4
))
print("OOF ROC-AUC:", round(
    roc_auc_score(y, oof_probabilities), 4
))
print("False alarms:", int(fp))
print("Missed failures:", int(fn))
print("True positives:", int(tp))
print("True negatives:", int(tn))


# --------------------------------------------------
# 6. Train the final candidate on all available rows
# --------------------------------------------------

final_model = create_model()
final_model.fit(X, y)


# --------------------------------------------------
# 7. Save a separate candidate bundle
# --------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_bundle = {
    "model": final_model,
    "model_name": "Random Forest",
    "threshold": threshold,
    "feature_names": list(X.columns),
    "target_name": "Machine failure",
    "selection_method": "5-fold out-of-fold threshold selection",
    "minimum_recall_target": MIN_RECALL,
    "random_state": RANDOM_STATE,
    "n_estimators": 200,
    "training_rows": int(len(X)),
}

joblib.dump(model_bundle, MODEL_PATH)

print("\nCANDIDATE MODEL SAVED")
print("---------------------")
print("Model:", model_bundle["model_name"])
print("Threshold:", model_bundle["threshold"])
print("Training rows:", model_bundle["training_rows"])
print("Path:", MODEL_PATH)
print(
    "\nImportant: OOF metrics are for threshold selection, "
    "not an independent final performance estimate."
)
