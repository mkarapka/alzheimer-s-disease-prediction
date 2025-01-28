from models.model_ import Model_
import xgboost as xgb
from sklearn.metrics import accuracy_score
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import numpy as np


class XGBoost(Model_):
    def __init__(self):
        self.cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        self.model = xgb.XGBClassifier()
        super().__init__(self.model)

    def objective(self, trial, X, y):

        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            # "tree_method": "gpu_hist",  # ✅ Włączenie GPU
        }

        accuracies = []

        # 🔹 Ręczna walidacja K-Fold
        for train_idx, val_idx in self.cv.split(X, y):
            X_train_fold, X_val_fold = (
                X[train_idx],
                X[val_idx],
            )
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Tworzenie modelu i trening
            model = xgb.XGBClassifier(eval_metric="auc", **params, n_jobs=-1)
            model.fit(X_train_fold, y_train_fold)

            # Predykcja i ocena
            y_pred = model.predict(X_val_fold)
            acc = accuracy_score(y_val_fold, y_pred)
            accuracies.append(acc)

        return np.mean(accuracies)  # Zwracamy średnią dokładność z walidacji krzyżowej
