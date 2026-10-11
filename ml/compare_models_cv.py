
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)


# --------------------------------------------------
# 1. Paths and settings
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

drop_columns = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

df = df.drop(columns=drop_columns)

df = pd.get_dummies(
    df,
    columns=["Type"],
    dtype=int,
)

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"].astype(int)

print("Dataset shape:", X.shape)
print("Failure examples:", int(y.sum()))
print("Normal examples:", int((y == 0).sum()))


# --------------------------------------------------
# 3. Define candidate models
# --------------------------------------------------

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=8,
        random_state=RANDOM_STATE,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_STATE,
    ),
}


# --------------------------------------------------
# 4. Create common cross-validation folds
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# --------------------------------------------------
# 5. Select a threshold from OOF probabilities
# --------------------------------------------------

def choose_threshold(y_true, probabilities):
    eligible_results = []
    all_results = []

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

        result = {
            "Threshold": float(threshold),
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
        }

        all_results.append(result)

        if recall >= MIN_RECALL:
            eligible_results.append(result)

    if eligible_results:
        # Highest precision, then highest F1,
        # then prefer the higher threshold.
        best = max(
            eligible_results,
            key=lambda row: (
                row["Precision"],
                row["F1-score"],
                row["Threshold"],
            ),
        )

        return best, True

    # If the recall target cannot be reached,
    # report the best available recall instead.
    best = max(
        all_results,
        key=lambda row: (
            row["Recall"],
            row["Precision"],
            row["F1-score"],
            row["Threshold"],
        ),
    )

    return best, False


# --------------------------------------------------
# 6. Compare models using out-of-fold predictions
# --------------------------------------------------

results = []

for name, model in models.items():
    print(f"\nEvaluating: {name}")

    # Every row receives a probability from a model
    # that did not train on that row's fold.
    oof_probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    best, target_met = choose_threshold(
        y.to_numpy(),
        oof_probabilities,
    )

    oof_predictions = (
        oof_probabilities >= best["Threshold"]
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        oof_predictions,
        labels=[0, 1],
    ).ravel()

    result = {
        "Model": name,
        "Recall Target Met": target_met,
        "Threshold": best["Threshold"],
        "Precision": best["Precision"],
        "Recall": best["Recall"],
        "F1-score": best["F1-score"],
        "Accuracy": accuracy_score(
            y, oof_predictions
        ),
        "Average Precision": average_precision_score(
            y, oof_probabilities
        ),
        "ROC-AUC": roc_auc_score(
            y, oof_probabilities
        ),
        "False Alarms": int(fp),
        "Missed Failures": int(fn),
        "True Positives": int(tp),
        "True Negatives": int(tn),
    }

    results.append(result)

    print("Recall target met:", target_met)
    print("Selected threshold:", best["Threshold"])
    print("Precision:", round(best["Precision"], 4))
    print("Recall:", round(best["Recall"], 4))
    print("F1-score:", round(best["F1-score"], 4))


# --------------------------------------------------
# 7. Display and save comparison report
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=[
        "Recall Target Met",
        "Precision",
        "F1-score",
    ],
    ascending=[False, False, False],
)

print("\nCROSS-VALIDATION COMPARISON")
print("---------------------------")
print(results_df.round(4).to_string(index=False))

REPORT_DIR.mkdir(parents=True, exist_ok=True)

report_path = REPORT_DIR / "model_comparison_cv.csv"
results_df.to_csv(report_path, index=False)

eligible_models = results_df[
    results_df["Recall Target Met"]
]

print("\nREPORT SAVED TO:", report_path)

if not eligible_models.empty:
    best_row = eligible_models.iloc[0]

    print("\nLEADING CANDIDATE")
    print("-----------------")
    print("Model:", best_row["Model"])
    print("Threshold:", best_row["Threshold"])
    print("Precision:", round(best_row["Precision"], 4))
    print("Recall:", round(best_row["Recall"], 4))
    print(
        "Average Precision:",
        round(best_row["Average Precision"], 4),
    )

    print(
        "\nNote: This is a development-stage candidate, "
        "not yet the final saved model."
    )
else:
    print(
        "\nWARNING: No candidate met the minimum "
        "80% recall target at the tested thresholds."
    )
    print(
        "Do not claim that the objective has been met. "
        "Review the models or revise the target."
    )

