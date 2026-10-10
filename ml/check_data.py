import pandas as pd

df = pd.read_csv("data/ai4i2020.csv")

print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nMachine Failure Distribution:")
print(df["Machine failure"].value_counts())