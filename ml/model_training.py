
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)


# 1. Load dataset
df = pd.read_csv("data/ai4i2020.csv")


# 2. Remove ID and failure-mode columns
drop_columns = [
    "UDI", "Product ID",
    "TWF", "HDF", "PWF", "OSF", "RNF"
]

df = df.drop(columns=drop_columns)


# 3. Convert Type into numerical columns
df = pd.get_dummies(
    df,
    columns=["Type"],
    dtype=int
)


# 4. Separate features and target
X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]


# 5. Create final test set (20%)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 6. Create training (60%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=42,
    stratify=y_train_full
)


# 7. Define models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
    ),

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


# 8. Threshold candidates
# Thresholds are selected ONLY using validation data.
thresholds = np.round(
    np.arange(0.05, 0.951, 0.01),
    2
)

validation_results = []
selected_models = {}


# 9. Train models and select thresholds
for name, model in models.items():

    model.fit(X_train, y_train)

    val_probabilities = model.predict_proba(X_val)[:, 1]

    # Calculate ranking metrics using probabilities
    val_pr_auc = average_precision_score(
        y_val, val_probabilities
    )

    val_roc_auc = roc_auc_score(
        y_val, val_probabilities
    )

    best_result = None

    for threshold in thresholds:

        val_predictions = (
            val_probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val,
            val_predictions,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            val_predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            val_predictions,
            zero_division=0
        )

        result = {
            "Model": name,
            "Threshold": float(threshold),
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "PR-AUC": val_pr_auc,
            "ROC-AUC": val_roc_auc
        }

        # Maximize validation F1.
        # If tied, prefer the higher threshold.
        if (
            best_result is None
            or result["F1-score"] > best_result["F1-score"]
            or (
                result["F1-score"] == best_result["F1-score"]
                and result["Threshold"] > best_result["Threshold"]
            )
        ):
            best_result = result

    validation_results.append(best_result)

    selected_models[name] = best_result["Threshold"]


# 10. Display validation comparison
validation_df = pd.DataFrame(validation_results)

print("\nVALIDATION COMPARISON")
print("---------------------")
print(validation_df.round(4).to_string(index=False))


# Select the leading model using validation F1 only.
# Test results will NOT be used to select the model.
best_validation_row = validation_df.loc[
    validation_df["F1-score"].idxmax()
]

best_model_name = best_validation_row["Model"]
best_threshold = float(best_validation_row["Threshold"])

print("\nMODEL SELECTED USING VALIDATION DATA")
print("------------------------------------")
print("Model:", best_model_name)
print("Threshold:", best_threshold)


# 11. Refit each model on training + validation data
# Evaluate on the reserved test set.
test_results = []
final_models = {}

for name, model in models.items():

    model.fit(X_train_full, y_train_full)

    test_probabilities = model.predict_proba(X_test)[:, 1]

    threshold = selected_models[name]

    test_predictions = (
        test_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    pr_auc = average_precision_score(
        y_test,
        test_probabilities
    )

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities
    )

    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    test_results.append({
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        "False Alarms": fp,
        "Missed Failures": fn
    })

    final_models[name] = model

    print(f"\n{name} - TEST RESULTS")
    print("---------------------")
    print("Threshold:", threshold)
    print("Confusion Matrix:")
    print(cm)


# 12. Display final test comparison
test_df = pd.DataFrame(test_results)

print("\nFINAL TEST COMPARISON")
print("---------------------")
print(test_df.round(4).to_string(index=False))


# 13. Save the model selected using validation F1
# This model has been refitted on training + validation data.
os.makedirs("models", exist_ok=True)

model_bundle = {
    "model": final_models[best_model_name],
    "model_name": best_model_name,
    "threshold": best_threshold,
    "feature_names": list(X.columns)
}

model_path = "models/predictive_maintenance_model.joblib"

joblib.dump(model_bundle, model_path)

print("\nFINAL MODEL SAVED")
print("-----------------")
print("Model:", best_model_name)
print("Threshold:", best_threshold)
print("Path:", model_path)
