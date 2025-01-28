import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import kagglehub
import os
from sklearn.model_selection import train_test_split
from functions.data_prep import data_preprocessing
from models.random_forest import Random_Forest_
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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

random_forest_model = Random_Forest_()
random_forest_model.bayesian_opt(X_train_scaled, y_train)

def Roc_():
    # Dopasowanie modelu z najlepszymi parametrami na danych treningowych
    random_forest_model.best_estimator.fit(X_train_scaled, y_train)

    # Użyj modelu do przewidywania prawdopodobieństw (nie klas)
    y_pred_prob = random_forest_model.best_estimator.predict_proba(X_test_scaled)

    # Oblicz krzywą ROC i AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1]) 
    roc_auc = auc(fpr, tpr)

    # Tworzenie wykresu AUC-ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Random Forest Model')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

Roc_()

