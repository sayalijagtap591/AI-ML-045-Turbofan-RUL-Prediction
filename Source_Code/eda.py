import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset columns
columns = (
    ["engine_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load training data
train_file = DATA_DIR / "train_FD004.txt"

if not train_file.exists():
    raise FileNotFoundError(
        f"Training file not found: {train_file}"
    )

data = pd.read_csv(
    train_file,
    sep=r"\s+",
    header=None,
    names=columns
)

print("\n========== DATASET INFORMATION ==========")
print("Dataset shape:", data.shape)
print("\nFirst five rows:")
print(data.head())

print("\n========== STATISTICAL SUMMARY ==========")
print(data.describe())

# 1. Operating cycles for each engine
engine_life = data.groupby("engine_id")["cycle"].max()

plt.figure(figsize=(10, 5))
plt.hist(engine_life, bins=20, edgecolor="black")
plt.title("Distribution of Engine Operating Cycles")
plt.xlabel("Operating Cycles")
plt.ylabel("Number of Engines")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "engine_life_distribution.png")
plt.close()

# 2. Sensor readings for one engine
engine_1 = data[data["engine_id"] == 1]

plt.figure(figsize=(10, 5))
plt.plot(engine_1["cycle"], engine_1["sensor_2"])
plt.title("Sensor 2 Readings for Engine 1")
plt.xlabel("Operating Cycle")
plt.ylabel("Sensor 2 Value")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sensor_2_engine_1.png")
plt.close()

# 3. Correlation between selected sensor readings
selected_sensors = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_11",
    "sensor_12",
    "sensor_15",
    "sensor_20",
    "sensor_21",
]

correlation = data[selected_sensors].corr()

plt.figure(figsize=(10, 8))
plt.imshow(correlation, aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(
    range(len(selected_sensors)),
    selected_sensors,
    rotation=45
)
plt.yticks(range(len(selected_sensors)), selected_sensors)
plt.title("Correlation Between Selected Sensors")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sensor_correlation.png")
plt.close()

print("\nEDA completed successfully!")
print("Graphs saved in:", OUTPUT_DIR)