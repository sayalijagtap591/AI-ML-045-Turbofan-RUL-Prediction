from flask import Flask, render_template, request
from pathlib import Path
import joblib
import json
import pandas as pd

app = Flask(
    __name__,
    template_folder="../templates"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

model_path = MODEL_DIR / "best_model.joblib"
features_path = MODEL_DIR / "feature_columns.json"

model = None
feature_columns = []

if model_path.exists() and features_path.exists():
    model = joblib.load(model_path)

    with open(features_path, "r", encoding="utf-8") as file:
        feature_columns = json.load(file)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        if model is None or not feature_columns:
            error = (
                "Model files are missing. "
                "Train the model before making predictions."
            )
        else:
            try:
                input_data = {}

                # Read all numerical input fields
                for feature in feature_columns:
                    raw_value = request.form.get(feature, "").strip()

                    if raw_value == "":
                        raise ValueError(
                            f"Please enter a value for {feature}."
                        )

                    input_data[feature] = float(raw_value)

                input_df = pd.DataFrame(
                    [input_data],
                    columns=feature_columns
                )

                prediction = max(
                    0.0,
                    float(model.predict(input_df)[0])
                )

            except ValueError as exc:
                error = str(exc)
            except Exception:
                error = (
                    "Prediction failed. Check the form fields "
                    "and trained model."
                )

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)