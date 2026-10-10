
import pandas as pd

from predict import predict_failure


# Load one actual record from the selected dataset
df = pd.read_csv("data/ai4i2020.csv")

row = df.iloc[0]

machine_data = {
    "Air temperature [K]": row["Air temperature [K]"],
    "Process temperature [K]": row["Process temperature [K]"],
    "Rotational speed [rpm]": row["Rotational speed [rpm]"],
    "Torque [Nm]": row["Torque [Nm]"],
    "Tool wear [min]": row["Tool wear [min]"],
    "Type": row["Type"]
}

result = predict_failure(machine_data)

print("Input source: AI4I dataset")
print("Actual dataset failure label:", int(row["Machine failure"]))
print("Prediction result:")

for key, value in result.items():
    print(f"{key}: {value}")

print("\nTesting invalid machine type...")

invalid_data = machine_data.copy()
invalid_data["Type"] = "X"

try:
    predict_failure(invalid_data)
except ValueError as error:
    print("Validation passed:", error)
else:
    print("ERROR: Invalid machine type was accepted")
