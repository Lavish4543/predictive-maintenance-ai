
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)


# --------------------------------------------------
# 1. Settings
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"
REPORT_DIR = BASE_DIR / "reports"

MIN_RECALL = 0.80
RANDOM_STATE = 42
N_SPLITS = 5

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


# --------------------------------------------------
# 2. Load and prepare data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

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


# --------------------------------------------------
# 3. Threshold selection helper
# --------------------------------------------------

def select_threshold(y_true, probabilities):
    eligible = []

    for threshold in THRESHOLDS:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if recall >= MIN_RECALL:
            eligible.append(
                {
                    "threshold": float(threshold),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

    if not eligible:
        return None

    return max(
        eligible,
        key=lambda item: (
            item["precision"],
            item["f1"],
            item["threshold"],
        ),
    )


# --------------------------------------------------
# 4. Nested threshold evaluation
# --------------------------------------------------

outer_cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

fold_results = []

for fold_number, (dev_idx, eval_idx) in enumerate(
    outer_cv.split(X, y),
    start=1,
):
    X_dev = X.iloc[dev_idx]
    y_dev = y.iloc[dev_idx]

    X_eval = X.iloc[eval_idx]
    y_eval = y.iloc[eval_idx]

    # Inner CV creates out-of-fold probabilities
    # using only this outer fold's development data.
    inner_cv = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE + fold_number,
    )

    inner_probabilities = np.zeros(len(X_dev))

    for train_idx, val_idx in inner_cv.split(
        X_dev, y_dev
    ):
        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(
            X_dev.iloc[train_idx],
            y_dev.iloc[train_idx],
        )

        inner_probabilities[val_idx] = (
            model.predict_proba(
                X_dev.iloc[val_idx]
            )[:, 1]
        )

    selected = select_threshold(
        y_dev.to_numpy(),
        inner_probabilities,
    )

    if selected is None:
        print(
            f"Fold {fold_number}: "
            "80% recall target not reached "
            "during threshold selection."
        )

        threshold = 0.50
        target_met = False
    else:
        threshold = selected["threshold"]
        target_met = True

    # Fit on the outer development fold only.
    final_fold_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    final_fold_model.fit(X_dev, y_dev)

    eval_probabilities = (
        final_fold_model.predict_proba(X_eval)[:, 1]
    )

    eval_predictions = (
        eval_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_eval,
        eval_predictions,
        labels=[0, 1],
    ).ravel()

    fold_results.append(
        {
            "Fold": fold_number,
            "Threshold": threshold,
            "Recall Target Met During Selection": target_met,
            "Precision": precision_score(
                y_eval,
                eval_predictions,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_eval,
                eval_predictions,
                zero_division=0,
            ),
            "F1-score": f1_score(
                y_eval,
                eval_predictions,
                zero_division=0,
            ),
            "Average Precision": average_precision_score(
                y_eval,
                eval_probabilities,
            ),
            "ROC-AUC": roc_auc_score(
                y_eval,
                eval_probabilities,
            ),
            "False Alarms": int(fp),
            "Missed Failures": int(fn),
            "True Positives": int(tp),
            "True Negatives": int(tn),
        }
    )

    print(
        f"Fold {fold_number}: "
        f"threshold={threshold:.2f}, "
        f"precision={fold_results[-1]['Precision']:.3f}, "
        f"recall={fold_results[-1]['Recall']:.3f}"
    )


# --------------------------------------------------
# 5. Summarize results
# --------------------------------------------------

results_df = pd.DataFrame(fold_results)

print("\nNESTED CROSS-VALIDATION RESULTS")
print("-------------------------------")
print(results_df.round(4).to_string(index=False))

print("\nMEAN METRICS ACROSS OUTER FOLDS")
print("--------------------------------")
metric_columns = [
    "Precision",
    "Recall",
    "F1-score",
    "Average Precision",
    "ROC-AUC",
]

print(results_df[metric_columns].mean().round(4))

print(
    "\nFolds meeting 80% recall during threshold selection:",
    int(
        results_df[
            "Recall Target Met During Selection"
        ].sum()
    ),
    "out of",
    N_SPLITS,
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

report_path = (
    REPORT_DIR / "random_forest_nested_cv.csv"
)

results_df.to_csv(report_path, index=False)

print("\nReport saved to:", report_path)
print(
    "\nThis is a more independent estimate of the "
    "threshold-selection procedure. It does not "
    "save or replace the production model."
)

