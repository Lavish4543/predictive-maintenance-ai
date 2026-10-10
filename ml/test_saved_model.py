
import joblib
import pandas as pd

# Load the saved model bundle
model_path = "models/predictive_maintenance_model.joblib"
bundle = joblib.load(model_path)

model = bundle["model"]
threshold = bundle["threshold"]
feature_names = bundle["feature_names"]

print("Model:", bundle["model_name"])
print("Threshold:", threshold)
print("Expected features:", feature_names)

# Read one real row from the selected dataset
df = pd.read_csv("data/ai4i2020.csv")

# Use only the input features expected by the model
drop_columns = [
    "UDI", "Product ID",
    "Machine failure",
    "TWF", "HDF", "PWF", "OSF", "RNF"
]

X = df.drop(columns=drop_columns)
X = pd.get_dummies(X, columns=["Type"], dtype=int)

# Ensure exact feature order and columns
X = X.reindex(columns=feature_names, fill_value=0)

sample = X.iloc[[0]]

# Predict failure probability
probability = model.predict_proba(sample)[0, 1]
prediction = int(probability >= threshold)

print("\nSaved model verification")
print("Failure probability:", round(float(probability), 4))
print("Selected threshold:", threshold)
print("Predicted class:", prediction)
print("Class meaning:", "Failure risk alert" if prediction == 1 else "No failure alert")
