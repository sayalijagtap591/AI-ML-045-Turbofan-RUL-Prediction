Turbofan Engine Degradation Prediction Using Machine Learning

1. Project Overview

This project aims to predict the Remaining Useful Life (RUL) of turbofan engines using machine learning.

Remaining Useful Life is the estimated number of operating cycles remaining before an engine reaches its end-of-life condition.

The project uses NASA's C-MAPSS turbofan engine simulation dataset, specifically the FD004 subset.

2. Objectives

- Analyze turbofan engine sensor data.
- Perform exploratory data analysis.
- Generate useful features from sensor readings.
- Train Lasso Regression and Ridge Regression models.
- Evaluate predictions using MAE and RMSE.
- Develop a Flask-based web interface for demonstration.

3. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Flask
- HTML
- CSS
- Joblib

4. Dataset

Dataset: NASA C-MAPSS Turbofan Engine Degradation Simulation Data

Subset: FD004

Files:

- train_FD004.txt
- test_FD004.txt
- RUL_FD004.txt

The dataset contains engine operating cycles, operating settings, and sensor measurements.

5. Machine Learning Models

Lasso Regression

Lasso Regression uses L1 regularization and can reduce the influence of less useful features.

Ridge Regression

Ridge Regression uses L2 regularization to reduce overfitting and stabilize model coefficients.

The model selection process compares validation performance using RMSE.

6. Project Structure

AI-ML-045_Turbofan_RUL_Project/
├── Source_Code/
│ ├── inspect_data.py
│ ├── eda.py
│ ├── feature_engineering.py
│ ├── train_models.py
│ ├── evaluate_model.py
│ └── app.py
├── data/
│ ├── train_FD004.txt
│ ├── test_FD004.txt
│ └── RUL_FD004.txt
├── models/
├── outputs/
├── templates/
│ └── index.html
├── requirements.txt
└── README.md

7. Installation

Create a Python virtual environment:

python -m venv .venv

Activate it in Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Install dependencies:

python -m pip install -r requirements.txt

8. Running the Project

Run the Python scripts in the following order:

1. Inspect the dataset.
2. Perform exploratory data analysis.
3. Generate training and testing features.
4. Train the machine learning models.
5. Evaluate the trained model.
6. Start the Flask application.

Start the web application:

python .\Source_Code\app.py

Open the local address displayed in the terminal.

9. Evaluation Metrics

The project uses:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Actual evaluation values should be recorded after successfully running the evaluation script.

10. Limitations

This is an educational machine learning prototype. Prediction quality depends on the trained model and the correctness of input features.

The demonstration application must not be used to make real aircraft maintenance decisions.

11. Future Enhancements

- Improve feature engineering.
- Compare additional regression models.
- Add prediction visualizations.
- Improve the web interface.
- Add CSV upload for batch predictions.
- Add automated model evaluation reports.

12. Conclusion

This project demonstrates a machine learning workflow for turbofan engine Remaining Useful Life prediction, from data analysis and feature engineering to model training, evaluation, and a web-based demonstration.
