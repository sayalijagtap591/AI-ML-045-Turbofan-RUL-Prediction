from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = BASE_DIR / "outputs" / "test_features.csv"
RUL_FILE = BASE_DIR / "data" / "RUL_FD004.txt"
MODEL_FILE = BASE_DIR / "models" / "best_model.joblib"
FEATURE_FILE = BASE_DIR / "models" / "feature_columns.json"

# Check required files
for file_path in [TEST_FILE, RUL_FILE, MODEL_FILE, FEATURE_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

# Load test data and trained model
test_data = pd.read_csv(TEST_FILE)
actual_rul_data = pd.read_csv(
    RUL_FILE, sep=r"\s+", header=None, names=["RUL"]
)
model = joblib.load(MODEL_FILE)

with open(FEATURE_FILE, "r", encoding="utf-8") as file:
    feature_columns = json.load(file)

# Select the last recorded cycle for each test engine
last_cycles = (
    test_data.sort_values(["engine_id", "cycle"])
    .groupby("engine_id", as_index=False)
    .tail(1)
    .sort_values("engine_id")
    .reset_index(drop=True)
)

# Prepare features in the same order used during training
X_test = last_cycles[feature_columns].replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

# Predict RUL for each engine
predicted_rul = model.predict(X_test)
predicted_rul = np.maximum(predicted_rul, 0)

# Actual RUL values correspond to test engines in ID order
actual_rul = actual_rul_data["RUL"].to_numpy()

if len(predicted_rul) != len(actual_rul):
    raise ValueError(
        f"Prediction count ({len(predicted_rul)}) does not match "
        f"actual RUL count ({len(actual_rul)})."
    )

# Calculate evaluation metrics
mae = mean_absolute_error(actual_rul, predicted_rul)
rmse = np.sqrt(mean_squared_error(actual_rul, predicted_rul))

# Create results table
results = pd.DataFrame({
    "engine_id": last_cycles["engine_id"].to_numpy(),
    "actual_RUL": actual_rul,
    "predicted_RUL": np.round(predicted_rul, 2),
})

# Save results
output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

results_file = output_dir / "rul_predictions.csv"
metrics_file = output_dir / "evaluation_metrics.json"

results.to_csv(results_file, index=False)

metrics = {
    "model": "Best trained model",
    "MAE": round(float(mae), 4),
    "RMSE": round(float(rmse), 4),
    "number_of_test_engines": int(len(results)),
}

with open(metrics_file, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\n========== RUL MODEL EVALUATION ==========")
print(f"Number of test engines: {len(results)}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

print("\nFirst 10 predictions:")
print(results.head(10).to_string(index=False))

print(f"\nPredictions saved to: {results_file}")
print(f"Metrics saved to: {metrics_file}")