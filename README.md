TURBO AI — Turbofan Engine Degradation Prediction

1. Project Overview

TURBO AI is a machine learning-based predictive maintenance project that estimates the Remaining Useful Life (RUL) of turbofan engines using sensor measurements and operating conditions.

The project uses the NASA C-MAPSS FD004 dataset and provides a web-based dashboard developed using Flask.

2. Project Objectives

- Estimate the Remaining Useful Life of a turbofan engine.
- Analyze sensor data to identify degradation patterns.
- Compare Lasso Regression and Ridge Regression.
- Display predictions through an interactive dashboard.
- Generate downloadable PDF prediction reports.
- Maintain a history of engine predictions.

3. Dataset

Dataset: NASA C-MAPSS FD004

The dataset contains engine operating cycles, three operational settings, and 21 sensor measurements.

The project uses the training data, test data, and test-set RUL values for model development and evaluation.

4. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Flask
- HTML and CSS
- JavaScript and Chart.js
- ReportLab
- NASA C-MAPSS dataset

5. Methodology

1. Load the training and testing datasets.
2. Inspect the data and handle preprocessing requirements.
3. Calculate Remaining Useful Life values for training engines.
4. Generate additional features from sensor measurements.
5. Calculate rolling means, rolling standard deviations, and sensor differences.
6. Train Lasso Regression and Ridge Regression models.
7. Compare model performance using evaluation metrics.
8. Select the model according to validation performance.
9. Integrate the trained model with a Flask application.
10. Display predictions, engine health categories, and maintenance guidance.
11. Generate PDF reports and save prediction history.

6. Machine Learning Models

Lasso Regression

Lasso Regression uses L1 regularization to penalize model coefficients and can reduce the influence of less useful features.

Ridge Regression

Ridge Regression uses L2 regularization to control model complexity and help reduce overfitting.

The final model should be identified using the actual model selection results generated during training.

7. Dashboard Features

- Engine ID and operating-cycle selection
- Remaining Useful Life prediction
- Engine health category
- Sensor analytics
- Maintenance guidance
- Prediction history
- Model performance information
- Downloadable PDF reports

8. Evaluation

Model performance should be reported using the actual values produced by the training and evaluation scripts.

Relevant metrics may include:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (R²)

Do not enter estimated or invented metric values.

9. Running the Application

Activate the project's virtual environment and install the required dependencies.

Run the Flask application from the project root:

python -u .\Source_Code\app.py

Open the following address in a browser:

http://127.0.0.1:5000

10. Limitations

This project is an educational machine learning prototype. Its predictions are estimates and must not be treated as certified engine-safety assessments.

Actual aircraft maintenance decisions require approved engineering procedures, validated models, and qualified technical personnel.

11. Conclusion

The project demonstrates how sensor data and regression models can be used to estimate turbofan engine Remaining Useful Life. The Flask dashboard provides a convenient interface for exploring predictions, reviewing sensor information, maintaining prediction history, and generating reports.