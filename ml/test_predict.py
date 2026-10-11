
import pandas as pd
from predict import predict_failure

df = pd.read_csv("data/ai4i2020.csv")
row = df.iloc[0]

print("DEBUG: Dataset values")
print("Type:", repr(row["Type"]))
print("Air temperature:", repr(row["Air temperature [K]"]))
print("Process temperature:", repr(row["Process temperature [K]"]))

machine_data = {
    "Air temperature [K]": row["Air temperature [K]"],
    "Process temperature [K]": row["Process temperature [K]"],
    "Rotational speed [rpm]": row["Rotational speed [rpm]"],
    "Torque [Nm]": row["Torque [Nm]"],
    "Tool wear [min]": row["Tool wear [min]"],
    "Type": row["Type"],
}

print("\nDEBUG: Input dictionary")
print(machine_data)

result = predict_failure(machine_data)

print("\nActual label:", int(row["Machine failure"]))
print("Prediction result:", result)

invalid_data = machine_data.copy()
invalid_data["Type"] = "X"

try:
    predict_failure(invalid_data)
except ValueError as error:
    print("Invalid input correctly rejected:", error)
else:
    raise AssertionError("Invalid machine type was not rejected.")

print("\nAll input tests passed.")
