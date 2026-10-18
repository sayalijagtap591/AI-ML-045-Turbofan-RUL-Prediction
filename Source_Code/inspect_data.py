import pandas as pd
from pathlib import Path

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset directory
DATA_DIR = PROJECT_DIR / "data"

# Dataset file paths
train_file = DATA_DIR / "train_FD004.txt"
test_file = DATA_DIR / "test_FD004.txt"
rul_file = DATA_DIR / "RUL_FD004.txt"

# Check whether files exist
for file in [train_file, test_file, rul_file]:
    if not file.exists():
        raise FileNotFoundError(f"Dataset file not found: {file}")

# Define column names
columns = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load datasets
train_data = pd.read_csv(
    train_file,
    sep=r"\s+",
    header=None,
    names=columns
)

test_data = pd.read_csv(
    test_file,
    sep=r"\s+",
    header=None,
    names=columns
)

rul_data = pd.read_csv(
    rul_file,
    sep=r"\s+",
    header=None,
    names=["RUL"]
)

# Display dataset information
print("\n========== TRAIN DATA ==========")
print("Shape:", train_data.shape)
print(train_data.head())

print("\n========== TEST DATA ==========")
print("Shape:", test_data.shape)
print(test_data.head())

print("\n========== RUL DATA ==========")
print("Shape:", rul_data.shape)
print(rul_data.head())

print("\n========== MISSING VALUES ==========")
print("Training data missing values:", train_data.isnull().sum().sum())
print("Testing data missing values:", test_data.isnull().sum().sum())

print("\nDataset loaded successfully!")