import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
import os
from others.logistic_regression import Logistic_Regression_clf, param_grid
from sklearn.model_selection import train_test_split
from models.others.svm import SVM_clf, param_grid_svm

# Download dataset
path = kagglehub.dataset_download("rabieelkharoua/alzheimers-disease-dataset")
files = os.listdir(path)

# Load CSV file
csv_file = files[0]
csv_path = os.path.join(path, csv_file)

# Load DataFrame
df = pd.read_csv(csv_path)
df = df.drop(columns=["DoctorInCharge"])


# Drop rows with missing values
X = np.array(df.drop(columns=["Diagnosis"]))
y = np.array(df["Diagnosis"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


log_reg = Logistic_Regression_clf()
X_train_scaled, X_test_scaled = log_reg.data_preprocessing(X_train, X_test)
# res = log_reg.gridsearch(X_train_scaled, y_train, param_grid, max_it=10000)
# results_df = pd.DataFrame(res)
# log_reg.show(results_df)

svm_clf = SVM_clf()
res = svm_clf.gridsearch(X_train_scaled, y_train, param_grid_svm)
print(res)

print(svm_clf.best_params)


# Chat
import optuna
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

# 1️⃣ Załaduj dane
X, y = load_iris(return_X_y=True)


# 2️⃣ Funkcja optymalizacyjna dla Bayesian Optimization
def objective(trial):
    C = trial.suggest_loguniform("C", 0.001, 100)  # Logarytmiczna skala dla C
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    degree = trial.suggest_int("degree", 2, 5)  # Tylko dla poly
    coef0 = trial.suggest_uniform("coef0", 0.0, 1.0)  # Dla poly i sigmoid
    shrinking = trial.suggest_categorical("shrinking", [True, False])
    tol = trial.suggest_loguniform("tol", 1e-4, 1e-2)

    # Tworzymy model SVM
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        shrinking=shrinking,
        tol=tol,
    )

    # Walidacja krzyżowa (5-fold)
    score = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
    return score  # Optymalizujemy pod kątem accuracy


# 3️⃣ Uruchamiamy optymalizację Bayesian
study = optuna.create_study(direction="maximize")  # Maksymalizujemy accuracy
study.optimize(objective, n_trials=50)  # 50 iteracji (można zwiększyć)

# 4️⃣ Najlepsze parametry
# print("✅ Najlepsze parametry:", study.best_params)
# print("📊 Najlepsza dokładność:", study.best_value)
