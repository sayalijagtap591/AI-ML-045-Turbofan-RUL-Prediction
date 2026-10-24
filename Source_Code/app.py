from flask import Flask, render_template, request
from pathlib import Path
from datetime import datetime
import json
import joblib
import pandas as pd
from io import BytesIO
from datetime import datetime

from flask import send_file, request
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__, template_folder="../templates")

# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "best_model.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"

TRAIN_PATH = OUTPUT_DIR / "train_features.csv"
TEST_PATH = OUTPUT_DIR / "test_features.csv"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"

HISTORY_PATH = OUTPUT_DIR / "prediction_history.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# LOAD MACHINE LEARNING MODEL
# ==================================================

model = None
feature_columns = []
model_error = None

try:
    if MODEL_PATH.exists() and FEATURES_PATH.exists():

        model = joblib.load(MODEL_PATH)

        with open(FEATURES_PATH, "r", encoding="utf-8") as file:
            feature_columns = json.load(file)

    else:
        model_error = (
            "Model files are missing. "
            "Please run train_models.py first."
        )

except Exception as exc:
    model_error = f"Model loading failed: {exc}"

# ==================================================
# LOAD DATASETS
# ==================================================

train_data = None
test_data = None
data_error = None

try:
    if TRAIN_PATH.exists():
        train_data = pd.read_csv(TRAIN_PATH)

    if TEST_PATH.exists():
        test_data = pd.read_csv(TEST_PATH)
    else:
        data_error = (
            "Test dataset is missing. "
            "Run feature_engineering.py first."
        )

except Exception as exc:
    data_error = f"Dataset loading failed: {exc}"

# ==================================================
# LOAD MODEL METRICS
# ==================================================

model_metrics = {}

try:
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            model_metrics = json.load(file)

except Exception as exc:
    model_metrics = {"error": str(exc)}

# ==================================================
# FEATURE 1: ENGINE HEALTH CLASSIFICATION
# ==================================================

def classify_engine_health(rul):

    if rul <= 30:
        return {
            "status": "CRITICAL",
            "level": "critical",
            "message": (
                "The predicted remaining useful life is low. "
                "Prioritize an engineering inspection."
            )
        }

    elif rul <= 80:
        return {
            "status": "WARNING",
            "level": "warning",
            "message": (
                "The predicted remaining useful life is decreasing. "
                "Schedule an engineering inspection."
            )
        }

    else:
        return {
            "status": "NORMAL",
            "level": "normal",
            "message": (
                "The predicted remaining useful life is above "
                "the configured warning threshold."
            )
        }

# ==================================================
# FEATURE 2: MAINTENANCE ADVISOR
# ==================================================

def get_maintenance_advice(rul, engine_row):

    advice = []

    # Recommendation based on predicted RUL
    if rul <= 30:

        advice.append({
            "priority": "HIGH",
            "title": "Priority Engineering Inspection",
            "description": (
                "Arrange an engineering assessment promptly. "
                "Do not use this prediction alone to decide "
                "whether an engine is safe to operate."
            )
        })

    elif rul <= 80:

        advice.append({
            "priority": "MEDIUM",
            "title": "Plan Preventive Maintenance",
            "description": (
                "Review the maintenance schedule and inspect "
                "the engine according to approved procedures."
            )
        })

    else:

        advice.append({
            "priority": "LOW",
            "title": "Continue Condition Monitoring",
            "description": (
                "Continue routine monitoring and follow the "
                "manufacturer's maintenance schedule."
            )
        })

    # Identify the largest sensor deviations against
    # the available processed test dataset.
    # These are screening indicators, not confirmed faults.

    if test_data is not None:

        sensor_columns = [
            column
            for column in engine_row.index
            if column.startswith("sensor_")
            and column.split("_")[1].isdigit()
        ]

        unusual_sensors = []

        for sensor in sensor_columns:

            if sensor not in test_data.columns:
                continue

            reference = pd.to_numeric(
                test_data[sensor],
                errors="coerce"
            ).dropna()

            if len(reference) < 10:
                continue

            value = pd.to_numeric(
                pd.Series([engine_row[sensor]]),
                errors="coerce"
            ).iloc[0]

            if pd.isna(value):
                continue

            mean = reference.mean()
            std = reference.std()

            if pd.notna(std) and std > 0:

                z_score = abs((value - mean) / std)

                if z_score >= 3:
                    unusual_sensors.append(
                        (sensor, round(float(z_score), 2))
                    )

        unusual_sensors.sort(
            key=lambda item: item[1],
            reverse=True
        )

        if unusual_sensors:

            sensor_names = ", ".join(
                sensor for sensor, _ in unusual_sensors[:5]
            )

            advice.append({
                "priority": "REVIEW",
                "title": "Sensor Readings Need Review",
                "description": (
                    f"Unusual readings were detected in: "
                    f"{sensor_names}. Compare them with approved "
                    f"operating limits and sensor trends."
                )
            })

        else:

            advice.append({
                "priority": "INFO",
                "title": "Sensor Screening",
                "description": (
                    "No sensor exceeded the configured three-standard-"
                    "deviation screening threshold. This does not "
                    "guarantee that the engine is fault-free."
                )
            })

    return advice

# ==================================================
# FEATURE 3: PREDICTION HISTORY
# ==================================================

def save_prediction_history(
    engine_id,
    cycle,
    rul,
    health_status
):

    record = pd.DataFrame([{
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "engine_id": engine_id,
        "cycle": cycle,
        "predicted_rul": round(float(rul), 2),
        "health_status": health_status
    }])

    record.to_csv(
        HISTORY_PATH,
        mode="a",
        header=not HISTORY_PATH.exists(),
        index=False
    )


def get_prediction_history():

    if not HISTORY_PATH.exists():
        return []

    try:
        history = pd.read_csv(HISTORY_PATH)

        return history.tail(10).iloc[::-1].to_dict(
            orient="records"
        )

    except Exception:
        return []

# ==================================================
# DATA ANALYSIS
# ==================================================

def get_data_analysis():

    analysis = {
        "train_rows": 0,
        "test_rows": 0,
        "train_engines": 0,
        "test_engines": 0,
        "sensor_count": 21,
        "feature_count": len(feature_columns),
        "max_cycle": 0,
        "engine_ids": [],
        "sensor_summary": [],
        "sensor_chart": {
            "labels": [],
            "datasets": []
        },
        "error": data_error
    }

    if train_data is not None:

        analysis["train_rows"] = len(train_data)

        if "engine_id" in train_data.columns:
            analysis["train_engines"] = int(
                train_data["engine_id"].nunique()
            )

    if test_data is None:
        return analysis

    required_columns = {"engine_id", "cycle"}

    if not required_columns.issubset(test_data.columns):

        analysis["error"] = (
            "The processed test dataset must contain "
            "engine_id and cycle columns."
        )

        return analysis

    analysis["test_rows"] = len(test_data)

    analysis["engine_ids"] = sorted(
        int(value)
        for value in test_data["engine_id"].dropna().unique()
    )

    analysis["test_engines"] = len(
        analysis["engine_ids"]
    )

    analysis["max_cycle"] = int(
        test_data["cycle"].max()
    )

    sensor_columns = [
        column
        for column in test_data.columns
        if column.startswith("sensor_")
        and column.split("_")[1].isdigit()
    ]

    analysis["sensor_count"] = len(sensor_columns)

    for sensor in sensor_columns:

        values = pd.to_numeric(
            test_data[sensor],
            errors="coerce"
        ).dropna()

        if not values.empty:

            analysis["sensor_summary"].append({
                "sensor": sensor,
                "mean": round(float(values.mean()), 3),
                "min": round(float(values.min()), 3),
                "max": round(float(values.max()), 3),
                "std": round(float(values.std()), 3)
            })

    selected_sensors = [
        f"sensor_{i}"
        for i in range(1, 6)
        if f"sensor_{i}" in test_data.columns
    ]

    if selected_sensors:

        grouped = (
            test_data.groupby("cycle")[selected_sensors]
            .mean()
            .sort_index()
        )

        if len(grouped) > 100:

            positions = [
                round(i * (len(grouped) - 1) / 99)
                for i in range(100)
            ]

            grouped = grouped.iloc[positions]

        analysis["sensor_chart"] = {
            "labels": [
                int(value)
                for value in grouped.index
            ],
            "datasets": [
                {
                    "label": sensor,
                    "data": [
                        round(float(value), 3)
                        if pd.notna(value)
                        else None
                        for value in grouped[sensor]
                    ]
                }
                for sensor in selected_sensors
            ]
        }

    return analysis

# ==================================================
# MAIN DASHBOARD
# ==================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = model_error

    selected_engine = None
    selected_cycle = None

    health = None
    maintenance_advice = []

    if request.method == "POST":

        try:

            if model is None or not feature_columns:

                raise ValueError(
                    model_error or "ML model is unavailable."
                )

            if test_data is None:

                raise ValueError(
                    data_error or "Test dataset is unavailable."
                )

            selected_engine = int(
                request.form.get("engine_id", "")
            )

            selected_cycle = int(
                request.form.get("cycle", "")
            )

            rows = test_data[
                (test_data["engine_id"] == selected_engine)
                & (test_data["cycle"] == selected_cycle)
            ]

            if rows.empty:

                raise ValueError(
                    "This engine ID and cycle combination "
                    "was not found in the test dataset."
                )

            missing_features = [
                column
                for column in feature_columns
                if column not in rows.columns
            ]

            if missing_features:

                raise ValueError(
                    "Missing model features: "
                    + ", ".join(missing_features[:5])
                )

            engine_row = rows.iloc[0]

            input_df = rows[feature_columns].apply(
                pd.to_numeric,
                errors="coerce"
            )

            if input_df.isnull().any().any():

                raise ValueError(
                    "Selected engine record contains missing "
                    "model feature values."
                )

            # Generate RUL prediction
            prediction = max(
                0.0,
                float(model.predict(input_df)[0])
            )

            # Classify engine health
            health = classify_engine_health(prediction)

            # Generate maintenance recommendations
            maintenance_advice = get_maintenance_advice(
                prediction,
                engine_row
            )

            # Store prediction in history
            save_prediction_history(
                selected_engine,
                selected_cycle,
                prediction,
                health["status"]
            )

            error = None

        except Exception as exc:

            error = str(exc)

    analysis = get_data_analysis()

    return render_template(
        "index.html",
        analysis=analysis,
        model_metrics=model_metrics,
        prediction=prediction,
        error=error,
        selected_engine=selected_engine,
        selected_cycle=selected_cycle,
        health=health,
        maintenance_advice=maintenance_advice,
        prediction_history=get_prediction_history()
    )

@app.route("/download-report", methods=["POST"])
def download_report():
    engine_id = request.form.get("engine_id", "")
    cycle = request.form.get("cycle", "")
    predicted_rul = request.form.get("predicted_rul", "")
    health_status = request.form.get("health_status", "Not available")

    try:
        engine_id = int(engine_id)
        cycle = int(cycle)
        predicted_rul = float(predicted_rul)

        if cycle < 1 or predicted_rul < 0:
            raise ValueError("Invalid prediction values.")

    except (ValueError, TypeError):
        return "Invalid report data. Please run a prediction again.", 400

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()
    content = []

    content.append(
        Paragraph("TURBO AI", styles["Title"])
    )
    content.append(
        Paragraph(
            "Turbofan Engine Health Intelligence Report",
            styles["Heading2"]
        )
    )
    content.append(Spacer(1, 18))

    content.append(
        Paragraph(
            "AI-ML-045 | NASA C-MAPSS FD004 Dataset",
            styles["Normal"]
        )
    )
    content.append(Spacer(1, 18))

    report_data = [
        ["Report Detail", "Value"],
        ["Generated On", datetime.now().strftime("%d-%m-%Y %H:%M:%S")],
        ["Engine ID", str(engine_id)],
        ["Operating Cycle", str(cycle)],
        ["Predicted Remaining Useful Life", f"{predicted_rul:.2f} cycles"],
        ["Engine Health Category", str(health_status)],
        ["Machine Learning Models", "Lasso Regression / Ridge Regression"],
    ]

    table = Table(report_data, colWidths=[210, 260])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123456")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("PADDING", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    content.append(table)
    content.append(Spacer(1, 20))

    content.append(
        Paragraph("Important Note", styles["Heading2"])
    )
    content.append(
        Paragraph(
            "This report contains an estimated Remaining Useful Life (RUL) "
            "generated by an educational machine-learning prototype. "
            "It is not a certified engine safety assessment. "
            "Actual maintenance decisions must follow approved engineering "
            "procedures and verified inspection results.",
            styles["BodyText"]
        )
    )
    
    content.append(Spacer(1, 20))
    content.append(
            Paragraph("Model Performance", styles["Heading2"])
    )

    metrics_path = BASE_DIR / "outputs" / "model_metrics.json"

    if metrics_path.exists():
        import json

        with open(metrics_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        metrics_data = [["Metric", "Recorded Value"]]

        for key, value in metrics.items():
            metrics_data.append([str(key), str(value)])

        metrics_table = Table(
            metrics_data,
            colWidths=[210, 260]
        )

        metrics_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123456")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))

        content.append(metrics_table)
    
    else:
        content.append(
            Paragraph(
                "Model metrics file was not found.",
                styles["BodyText"]
            )
        )
    document.build(content)
    content.append(Spacer(1, 20))
    content.append(
        Paragraph("Sensor Analytics", styles["Heading2"])
    )

    sensor_data = [
        ["Sensor", "Mean", "Minimum", "Maximum", "Std. Deviation"]
    ]

    # Read sensor statistics from the application's analysis data.
    # This section uses a saved sensor summary when available.
    sensor_summary_path = BASE_DIR / "outputs" / "sensor_summary.csv"

    if sensor_summary_path.exists():
        import pandas as pd

        sensor_df = pd.read_csv(sensor_summary_path)

        for _, row in sensor_df.iterrows():
            sensor_data.append([
                str(row.get("sensor", "")),
                str(row.get("mean", "")),
                str(row.get("min", "")),
                str(row.get("max", "")),
                str(row.get("std", "")),
            ])

        sensor_table = Table(
            sensor_data,
            repeatRows=1,
            colWidths=[75, 90, 90, 90, 110],
        )

        sensor_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123456")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))

        content.append(sensor_table)
    else:
        content.append(
            Paragraph(
                "Sensor summary file is not available. "
                "Sensor statistics have not been added to this report.",
                styles["BodyText"],
            )
        )
    buffer.seek(0)

    filename = f"TURBO_AI_Engine_{engine_id}_Report.pdf"

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename
    )
# ==================================================
# RUN APPLICATION
# ==================================================

if __name__ == "__main__":

    print("Starting TURBO AI Dashboard...")

    print(
        "Model loaded:",
        model is not None
    )

    print(
        "Test dataset loaded:",
        test_data is not None
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
