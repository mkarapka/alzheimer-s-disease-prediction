import numpy as np


def confiusion_matrix(y, y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0
    for y_i, y_p_i in zip(y, y_pred):
        if y_i == 1:
            if y_p_i == 1:
                TP += 1
            else:
                FN += 1
        else:
            if y_p_i == 1:
                FP += 1
            else:
                TN += 1
    return np.array([[TP, FP], [FN, TN]])


def accuracy(y: np.array, y_pred: np.array) -> float:
    return np.sum(y == y_pred) / y.shape[0]


def recall(y: np.array, y_pred: np.array) -> float:
    TP = np.sum((y == 1) and (y_pred == 1))
    FN = np.sum((y == 1) and (y_pred == 0))
    return TP / (TP + FN)


def precision(y, y_pred):
    TP = np.sum((y == 1) and (y_pred == 1))
    FP = np.sum((y == 0) and (y_pred == 1))
    return TP / (TP + FP)


def f1_score(y, y_pred):
    prsn = precision(y, y_pred)
    recll = recall(y, y_pred)
    return 2.0 * prsn * recll / (prsn + recll)


def roc_auc(y, y_pred):
    pass


y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
y_pred = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 0])
print(accuracy(y, y_pred))
