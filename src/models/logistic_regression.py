from models.model_ import Model_
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from functions.visualization import plot_optimization_results, plot_best_parameters
import optuna


class Logistic_Regression_(Model_):
    def __init__(self):
        self.model = LogisticRegression()
        super().__init__(self.model)

    def objective(self, trial, X, y):
        solver_choices = ["liblinear", "saga", "newton-cg", "sag", "lbfgs"]
        penalty_choices = ["l1", "l2", "none"]  # Pełna lista możliwości

        # 1️⃣ Sugerujemy solver
        solver = trial.suggest_categorical("solver", solver_choices)

        # 2️⃣ Sugerujemy penalty (bez dynamicznych zmian)
        penalty = trial.suggest_categorical("penalty", penalty_choices)

        # 3️⃣ Wybór parametru C
        C = trial.suggest_float("C", 0.001, 100, log=True)

        # 4️⃣ Filtrujemy niedozwolone kombinacje
        invalid_combinations = (
            (solver in ["newton-cg", "sag", "lbfgs"] and penalty != "l2")
            or (solver == "liblinear" and penalty not in ["l1", "l2"])
            or (solver == "saga" and penalty not in ["l1", "l2"])
        )

        if invalid_combinations:
            raise optuna.TrialPruned()  # Odrzucamy niepoprawne kombinacje

        model = LogisticRegression(
            solver=solver, penalty=penalty if penalty != "none" else None, C=C
        )
        score = cross_val_score(model, X, y, cv=5, scoring="recall").mean()

        return score

    def visualize_results(self, study):
        plot_optimization_results(study, title="Logistic Regression Optimization")
        plot_best_parameters(study, title="Logistic Regression Best Parameters")
