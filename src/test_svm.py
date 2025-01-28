import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import kagglehub
import os
from sklearn.model_selection import train_test_split
from functions.data_prep import data_preprocessing
from models.svm import SVM
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

path = kagglehub.dataset_download("rabieelkharoua/alzheimers-disease-dataset")
files = os.listdir(path)
print("Content of", files)

csv_file = files[0]
csv_path = os.path.join(path, csv_file)

# Load DataFrame
df = pd.read_csv(csv_path)
df = df.drop(columns=["DoctorInCharge"])  # Drop useless column


# Display the content of DataFrame
df.head().T

X = np.array(df.drop(columns=["Diagnosis"]))
y = np.array(df["Diagnosis"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_scaled, X_test_scaled = data_preprocessing(X_train, X_test)

svm_model = SVM()
svm_model.bayesian_opt(X_train_scaled, y_train)

def Roc_():
    # Używamy najlepszego modelu z bayesian_opt
    best_model = svm_model.best_estimator.fit(X_train_scaled, y_train)

    # Przewidywania dla danych testowych
    # Zamiast y_score = best_model.predict(X_test_scaled)
    y_score = best_model.decision_function(X_test_scaled)

    # Jeśli mamy więcej niż 2 klasy, używamy binarnej wersji etykiet
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

    # Oblicz krzywą ROC
    fpr, tpr, thresholds = roc_curve(y_test_bin, y_score)
    roc_auc = auc(fpr, tpr)

    # Wykres AUC ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="b", label=f"ROC curve (AUC = {roc_auc})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for SVM Model")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

Roc_()