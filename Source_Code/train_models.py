import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = OUTPUT_DIR / "train_features.csv"
TEST_FILE = OUTPUT_DIR / "test_features.csv"


# ==========================================
# 2. CHECK FILES
# ==========================================

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training features not found: {TRAIN_FILE}. "
        "Run feature_engineering.py first."
    )

if not TEST_FILE.exists():
    raise FileNotFoundError(
        f"Testing features not found: {TEST_FILE}. "
        "Run feature_engineering.py first."
    )


# ==========================================
# 3. LOAD DATA
# ==========================================

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)

print("Training dataset shape:", train_data.shape)
print("Testing dataset shape:", test_data.shape)


# ==========================================
# 4. DEFINE FEATURES AND TARGET
# ==========================================

excluded_columns = [
    "engine_id",
    "RUL"
]

feature_columns = [
    column
    for column in train_data.columns
    if column not in excluded_columns
]

# Use the same features in training and testing.
missing_features = [
    column
    for column in feature_columns
    if column not in test_data.columns
]

if missing_features:
    raise ValueError(
        f"Missing test features: {missing_features}"
    )

X = train_data[feature_columns].replace(
    [np.inf, -np.inf], np.nan
)

X = X.fillna(0)

y = train_data["RUL"]

groups = train_data["engine_id"]


# ==========================================
# 5. SPLIT DATA BY ENGINE
# ==========================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, val_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices]
X_val = X.iloc[val_indices]

y_train = y.iloc[train_indices]
y_val = y.iloc[val_indices]

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_val))

print(
    "Training engines:",
    groups.iloc[train_indices].nunique()
)

print(
    "Validation engines:",
    groups.iloc[val_indices].nunique()
)


# ==========================================
# 6. DEFINE MODELS
# ==========================================

models = {
    "Lasso": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1, max_iter=10000))
    ]),

    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
}


# ==========================================
# 7. TRAIN AND EVALUATE
# ==========================================

results = {}
trained_models = {}

for model_name, model in models.items():

    print(f"\nTraining {model_name} Regression...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    mae = mean_absolute_error(
        y_val,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            predictions
        )
    )

    results[model_name] = {
        "MAE": float(mae),
        "RMSE": float(rmse)
    }

    trained_models[model_name] = model

    print(f"{model_name} MAE: {mae:.4f}")
    print(f"{model_name} RMSE: {rmse:.4f}")


# ==========================================
# 8. SELECT BEST MODEL
# ==========================================

best_model_name = min(
    results,
    key=lambda name: results[name]["RMSE"]
)

best_model = trained_models[best_model_name]

print("\nBest validation model:", best_model_name)


# ==========================================
# 9. SAVE MODELS
# ==========================================

for model_name, model in trained_models.items():

    model_path = MODEL_DIR / f"{model_name.lower()}_model.joblib"

    joblib.dump(model, model_path)

    print(f"Saved {model_name} model:", model_path)


best_model_path = MODEL_DIR / "best_model.joblib"

joblib.dump(best_model, best_model_path)

# Save feature names for Flask predictions.
feature_info_path = MODEL_DIR / "feature_columns.json"

with open(feature_info_path, "w", encoding="utf-8") as file:
    json.dump(feature_columns, file, indent=2)


# ==========================================
# 10. SAVE METRICS
# ==========================================

metrics_path = OUTPUT_DIR / "model_metrics.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(
        {
            "validation_metrics": results,
            "best_model": best_model_name,
            "feature_count": len(feature_columns)
        },
        file,
        indent=2
    )


# ==========================================
# 11. FINAL SUMMARY
# ==========================================

print("\n========== MODEL TRAINING SUMMARY ==========")

for model_name, metrics in results.items():

    print(f"\n{model_name}")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")

print("\nBest model:", best_model_name)
print("Feature count:", len(feature_columns))
print("Models saved in:", MODEL_DIR)
print("Metrics saved in:", metrics_path)

print("\nModel training completed successfully!")