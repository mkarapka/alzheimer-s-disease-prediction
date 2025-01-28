from .model_ import Model_
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from functions.visualization import plot_optimization_results, plot_best_parameters


class SVM(Model_):
    def __init__(self):
        self.model = SVC()
        super().__init__(self.model)

    def objective(self, trial, X, y):
        C = trial.suggest_float("C", 0.001, 100, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        kernel = trial.suggest_categorical(
            "kernel", ["linear", "poly", "rbf", "sigmoid"]
        )
        degree = trial.suggest_int("degree", 2, 5)  # Only for poly
        coef0 = trial.suggest_float("coef0", 0.0, 1.0)  # Only for poly and sigmoid

        model = SVC(C=C, gamma=gamma, kernel=kernel, degree=degree, coef0=coef0)
        scores = cross_val_score(model, X, y, cv=5, scoring="recall")
        return scores.mean()

    def visualize_results(self, study):
        plot_optimization_results(study, title="SVM Optimization")
        plot_best_parameters(study, title="SVM Best Parameters")
