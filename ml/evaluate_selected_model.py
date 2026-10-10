
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score
)

# Load and prepare the dataset
df = pd.read_csv("data/ai4i2020.csv")

drop_columns = [
    "UDI", "Product ID",
    "TWF", "HDF", "PWF", "OSF", "RNF"
]

df = df.drop(columns=drop_columns)
df = pd.get_dummies(df, columns=["Type"], dtype=int)

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

# Recreate the exact same data splits
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.25,
    random_state=42,
    stratify=y_train_full
)

# Fit Random Forest using training data only
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Select threshold using validation data only
val_probabilities = model.predict_proba(X_val)[:, 1]

thresholds = [round(x, 2) for x in
              __import__("numpy").arange(0.05, 0.951, 0.01)]

from sklearn.metrics import precision_score, recall_score

eligible = []

for threshold in thresholds:
    val_predictions = (val_probabilities >= threshold).astype(int)

    precision = precision_score(
        y_val, val_predictions, zero_division=0
    )
    recall = recall_score(
        y_val, val_predictions, zero_division=0
    )

    if recall >= 0.80:
        eligible.append((precision, threshold))

if not eligible:
    raise RuntimeError(
        "Validation data par 80% recall target achieve nahi hua."
    )

# Highest precision among thresholds meeting the recall target
best_precision, selected_threshold = max(
    eligible, key=lambda item: (item[0], item[1])
)

# Refit on training + validation data
model.fit(X_train_full, y_train_full)

# Evaluate once on the untouched test set
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (
    test_probabilities >= selected_threshold
).astype(int)

print("\nSELECTED VALIDATION THRESHOLD")
print("Threshold:", selected_threshold)
print("Validation precision:", round(best_precision, 4))

print("\nFINAL TEST RESULTS")
print("Confusion matrix:")
print(confusion_matrix(y_test, test_predictions, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_test,
    test_predictions,
    target_names=["Normal", "Failure"],
    zero_division=0
))

print("Test PR-AUC (Average Precision):",
      round(average_precision_score(y_test, test_probabilities), 4))

print("Test ROC-AUC:",
      round(roc_auc_score(y_test, test_probabilities), 4))

# Save the final evaluated model and selected threshold
os.makedirs("models", exist_ok=True)

model_bundle = {
    "model": model,
    "model_name": "Random Forest",
    "threshold": selected_threshold,
    "feature_names": list(X.columns)
}

model_path = "models/predictive_maintenance_model.joblib"

joblib.dump(model_bundle, model_path)

print("\nFINAL MODEL SAVED")
print("Model: Random Forest")
print("Threshold:", selected_threshold)
print("Path:", model_path)
