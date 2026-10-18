import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train_FD004.txt"
TEST_FILE = DATA_DIR / "test_FD004.txt"
RUL_FILE = DATA_DIR / "RUL_FD004.txt"

# ==========================================
# 2. DATASET COLUMN NAMES
# ==========================================

columns = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

sensor_columns = [
    f"sensor_{i}" for i in range(1, 22)
]

# ==========================================
# 3. CHECK DATASET FILES
# ==========================================

for file in [TRAIN_FILE, TEST_FILE, RUL_FILE]:
    if not file.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file}"
        )

# ==========================================
# 4. LOAD DATASETS
# ==========================================

train_data = pd.read_csv(
    TRAIN_FILE,
    sep=r"\s+",
    header=None,
    names=columns
)

test_data = pd.read_csv(
    TEST_FILE,
    sep=r"\s+",
    header=None,
    names=columns
)

rul_data = pd.read_csv(
    RUL_FILE,
    sep=r"\s+",
    header=None,
    names=["RUL"]
)

print("Training shape:", train_data.shape)
print("Testing shape:", test_data.shape)
print("RUL shape:", rul_data.shape)

# ==========================================
# 5. GENERATE TRAINING RUL LABELS
# ==========================================

max_cycles = train_data.groupby(
    "engine_id"
)["cycle"].transform("max")

train_data["RUL"] = max_cycles - train_data["cycle"]

print("\nTraining RUL labels generated.")

# ==========================================
# 6. GENERATE TESTING RUL LABELS
# ==========================================

# RUL file contains the remaining cycles at the
# final observed cycle of each test engine.

test_engine_ids = sorted(
    test_data["engine_id"].unique()
)

if len(test_engine_ids) != len(rul_data):
    raise ValueError(
        "Number of test engines does not match RUL labels."
    )

test_rul_map = dict(
    zip(test_engine_ids, rul_data["RUL"])
)

test_last_cycles = test_data.groupby(
    "engine_id"
)["cycle"].max()

test_data["RUL"] = (
    test_data["engine_id"].map(test_rul_map)
    + test_data["engine_id"].map(test_last_cycles)
    - test_data["cycle"]
)

print("Testing RUL labels generated.")

# ==========================================
# 7. ROLLING-WINDOW FEATURE ENGINEERING
# ==========================================

WINDOW_SIZE = 5


def create_rolling_features(data):

    data = data.sort_values(
        ["engine_id", "cycle"]
    ).copy()

    grouped = data.groupby("engine_id", sort=False)

    for sensor in sensor_columns:

        # Rolling mean over current and previous cycles
        data[f"{sensor}_roll_mean"] = grouped[
            sensor
        ].transform(
            lambda series: series.rolling(
                window=WINDOW_SIZE,
                min_periods=1
            ).mean()
        )

        # Rolling standard deviation
        data[f"{sensor}_roll_std"] = grouped[
            sensor
        ].transform(
            lambda series: series.rolling(
                window=WINDOW_SIZE,
                min_periods=2
            ).std()
        )

    # Standard deviation is undefined for a single reading.
    rolling_std_columns = [
        f"{sensor}_roll_std"
        for sensor in sensor_columns
    ]

    data[rolling_std_columns] = data[
        rolling_std_columns
    ].fillna(0)

    return data


train_data = create_rolling_features(train_data)
test_data = create_rolling_features(test_data)

# ==========================================
# 8. HANDLE INVALID VALUES
# ==========================================

train_data.replace(
    [np.inf, -np.inf], np.nan, inplace=True
)

test_data.replace(
    [np.inf, -np.inf], np.nan, inplace=True
)

# ==========================================
# 9. SAVE PROCESSED DATASETS
# ==========================================

train_output = OUTPUT_DIR / "train_features.csv"
test_output = OUTPUT_DIR / "test_features.csv"

train_data.to_csv(train_output, index=False)
test_data.to_csv(test_output, index=False)

# ==========================================
# 10. DISPLAY RESULTS
# ==========================================

print("\n========== FEATURE ENGINEERING RESULTS ==========")

print("Training rows:", len(train_data))
print("Training columns:", len(train_data.columns))

print("Testing rows:", len(test_data))
print("Testing columns:", len(test_data.columns))

print("\nSample training data:")
print(
    train_data[
        ["engine_id", "cycle", "sensor_2", "RUL",
         "sensor_2_roll_mean", "sensor_2_roll_std"]
    ].head(10)
)

print("\nTraining RUL range:")
print(train_data["RUL"].min(), "to", train_data["RUL"].max())

print("\nSaved files:")
print(train_output)
print(test_output)

print("\nFeature engineering completed successfully!")