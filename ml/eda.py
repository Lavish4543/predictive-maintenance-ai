import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/ai4i2020.csv")

# print("Basic Statistics:")
# print(df.describe())
# print("\nAverage values by Machine Failure:")
# print(
#     df.groupby("Machine failure")[
#         [
#             "Air temperature [K]",
#             "Process temperature [K]",
#             "Rotational speed [rpm]",
#             "Torque [Nm]",
#             "Tool wear [min]"
#         ]
#     ].mean()
# )
# print("\nFailure vs Normal - All Features:")

# features = [
#     "Air temperature [K]",
#     "Process temperature [K]",
#     "Rotational speed [rpm]",
#     "Torque [Nm]",
#     "Tool wear [min]"
# ]

# comparison = df.groupby("Machine failure")[features].mean()

# print(comparison)


# comparison.plot(kind="bar", figsize=(10, 6))

# plt.title("Normal vs Failure - Average Sensor Values")
# plt.xlabel("Machine Failure")
# plt.ylabel("Average Value")
# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.show()

# # Feature distributions

# features = [
#     "Air temperature [K]",
#     "Process temperature [K]",
#     "Rotational speed [rpm]",
#     "Torque [Nm]",
#     "Tool wear [min]"
# ]

# for feature in features:
#     plt.figure(figsize=(8, 5))

#     df[df["Machine failure"] == 0][feature].hist(
#         alpha=0.6,
#         label="Normal"
#     )

#     df[df["Machine failure"] == 1][feature].hist(
#         alpha=0.6,
#         label="Failure"
#     )

#     plt.title(f"{feature} - Normal vs Failure")
#     plt.xlabel(feature)
#     plt.ylabel("Number of Records")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()



print("\nCorrelation with Machine Failure:")

correlation = df[
    [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure"
    ]
].corr()

print(correlation["Machine failure"].sort_values(ascending=False))

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nProduct Type Values:")
print(df["Type"].value_counts())