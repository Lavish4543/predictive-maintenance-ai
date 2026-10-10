
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)


# 1. Load and prepare the same dataset
df = pd.read_csv("data/ai4i2020.csv")

drop_columns = [
    "UDI", "Product ID",
    "TWF", "HDF", "PWF", "OSF", "RNF"
]

df = df.drop(columns=drop_columns)

df = pd.get_dummies(
    df,
    columns=["Type"],
    dtype=int
)

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]


# 2. Recreate the same train-validation-test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=42,
    stratify=y_train_full
)


# 3. Define the two candidate models
models = {
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=8,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# 4. Set the operational recall target
TARGET_RECALL = 0.80

# Evaluate many thresholds on validation data only.
thresholds = np.round(
    np.arange(0.05, 0.951, 0.01),
    2
)

all_results = []


# 5. Fit models and analyze validation thresholds
for name, model in models.items():

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]

    pr_auc = average_precision_score(
        y_val,
        probabilities
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1]
        ).ravel()

        all_results.append({
            "Model": name,
            "Threshold": float(threshold),
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "PR-AUC": pr_auc,
            "False Alarms": int(fp),
            "Missed Failures": int(fn),
            "Detected Failures": int(tp)
        })


# 6. Store all validation results
results_df = pd.DataFrame(all_results)

results_df.to_csv(
    "models/threshold_validation_results.csv",
    index=False
)


# 7. Show the best F1 result for each model
print("\nBEST F1 THRESHOLD PER MODEL")
print("----------------------------")

for name in models:

    model_results = results_df[
        results_df["Model"] == name
    ]

    best_row = model_results.loc[
        model_results["F1-score"].idxmax()
    ]

    print(
        f"\n{name}: "
        f"Threshold={best_row['Threshold']:.2f}, "
        f"Precision={best_row['Precision']:.4f}, "
        f"Recall={best_row['Recall']:.4f}, "
        f"F1={best_row['F1-score']:.4f}, "
        f"False Alarms={best_row['False Alarms']}, "
        f"Missed Failures={best_row['Missed Failures']}"
    )


# 8. Filter thresholds that meet the recall target
eligible = results_df[
    results_df["Recall"] >= TARGET_RECALL
].copy()

print(f"\nTHRESHOLDS MEETING RECALL >= {TARGET_RECALL:.0%}")
print("--------------------------------------")

if eligible.empty:

    print(
        "Neither model achieved the target "
        "on the tested validation thresholds."
    )

else:

    # Among eligible thresholds, prioritize precision.
    # Break ties using F1, then fewer false alarms.
    eligible = eligible.sort_values(
        by=[
            "Precision",
            "F1-score",
            "False Alarms"
        ],
        ascending=[False, False, True]
    )

    best_per_model = (
        eligible
        .groupby("Model", sort=False)
        .head(1)
    )

    print(
        best_per_model[
            [
                "Model",
                "Threshold",
                "Precision",
                "Recall",
                "F1-score",
                "PR-AUC",
                "False Alarms",
                "Missed Failures"
            ]
        ].round(4).to_string(index=False)
    )


print("\nFull validation results saved to:")
print("models/threshold_validation_results.csv")
