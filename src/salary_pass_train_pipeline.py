import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ----------------------------------------
# MLflow Configuration
# ----------------------------------------
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_TRACKING_URI", "http://10.145.40.9:5000")
)

# ----------------------------------------
# MODEL 1: Highest Salary Model
# ----------------------------------------
def train_salary_model():

    mlflow.set_experiment("Salaryandpass")

    data = {
        "Experience": [1, 3, 5, 7, 10, 2, 6, 8],
        "Age": [22, 24, 28, 32, 38, 25, 30, 35],
        "Education_Level": ["Bachelors","Bachelors","Masters","Masters","PhD","Bachelors","Bachelors","PhD"],
        "Department": ["HR","Sales","IT","IT","R&D","Sales","HR","R&D"],
        "Salary": [28000, 35000, 60000, 75000, 120000, 32000, 45000, 110000]
    }

    df = pd.DataFrame(data)
    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop("Salary", axis=1)
    y = df_encoded["Salary"]

    model = LinearRegression()

    with mlflow.start_run(run_name="salary-model-training"):
        model.fit(X, y)
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)

        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="HighestSalaryModel"
        )

        print("✅ HighestSalaryModel Registered")


# ----------------------------------------
# MODEL 2: Student Pass Percentage Model
# ----------------------------------------
def train_student_pass_model():

    mlflow.set_experiment("StudentPassModel")

    student_data = {
        "Gender": ["Boys","Girls","Boys","Girls"],
        "Total_Students": [1000, 950, 1100, 1000],
        "Passed_Students": [850, 900, 990, 950]
    }

    df = pd.DataFrame(student_data)
    df["Pass_Percentage"] = (df["Passed_Students"] / df["Total_Students"]) * 100
    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop("Pass_Percentage", axis=1)
    y = df_encoded["Pass_Percentage"]

    model = LinearRegression()

    with mlflow.start_run(run_name="student-pass-training"):
        model.fit(X, y)
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)

        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="StudentPassModel"
        )

        print("✅ StudentPassModel Registered")


# ----------------------------------------
# PIPELINE ENTRY POINT
# ----------------------------------------
if __name__ == "__main__":
    train_salary_model()
    train_student_pass_model()

