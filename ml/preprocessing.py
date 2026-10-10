import pandas as pd
from sklearn.model_selection import train_test_split

# Dataset load
df = pd.read_csv("data/ai4i2020.csv")

# Columns that are not useful as model inputs
drop_columns = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]

df = df.drop(columns=drop_columns)

# Convert Type (L, M, H) into numerical columns
df = pd.get_dummies(df, columns=["Type"], dtype=int)

# Separate features and target
X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nFeatures used by the model:")
print(X.columns.tolist()) 