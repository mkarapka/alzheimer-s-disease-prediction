import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def confiusion_matrix(y: np.array, y_pred: np.array) -> np.array:
    TP = np.sum((y == 1) and (y_pred == 1))
    TN = np.sum(((y == 0) and (y_pred == 0)))
    FP = np.sum((y == 0) and (y_pred == 1))
    FN = np.sum((y == 1) and (y_pred == 0))
    return np.array([[TP, FP], [FN, TN]])


def accuracy(y: np.array, y_pred: np.array) -> float:
    return np.sum(y == y_pred) / y.shape[0]


def recall(y: np.array, y_pred: np.array) -> float:
    TP = np.sum((y == 1) and (y_pred == 1))
    FN = np.sum((y == 1) and (y_pred == 0))
    return TP / (TP + FN)


def precision(y: np.array, y_pred: np.array) -> float:
    TP = np.sum((y == 1) and (y_pred == 1))
    FP = np.sum((y == 0) and (y_pred == 1))
    return TP / (TP + FP)


def f1_score(y: np.array, y_pred: np.array) -> float:
    prsn = precision(y, y_pred)
    recll = recall(y, y_pred)
    return 2.0 * prsn * recll / (prsn + recll)


def roc_auc(y, y_pred):
    def auc(x, y):
        auc = 0
        for i in range(1, len(x)):
            auc += (x[i] - x[i - 1]) * y[i]
        return auc

    FPR, TPR, thresholds = roc_curve(y, y_pred)
    return FPR, TPR, auc(FPR, TPR)


# test
y = np.array([0, 0, 1, 1])  # Prawdziwe etykiety
y_pred = np.array([0.1, 0.4, 0.35, 0.8])  # Przewidywane prawdopodobieństwa


FPR, TPR, AUC = roc_auc(y, y_pred)
plt.plot(FPR, TPR, marker="o", label=f"AUC = {AUC}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
