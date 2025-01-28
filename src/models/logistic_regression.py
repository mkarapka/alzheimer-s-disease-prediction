from models.model_ import Model_
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from visualization import plot_optimization_results, plot_best_parameters


class Logistic_Regression_(Model_):
    def __init__(self):
        self.model = LogisticRegression()
        super().__init__(self.model)

    def objective(self, trial, X, y):
        solver = trial.suggest_categorical(
            "solver", ["liblinear", "saga", "newton-cg", "sag", "lbfgs"]
        )

        # Define all possible penalties
        all_penalties = ["l1", "l2", None]

        # Suggest penalty from all possible penalties
        penalty = trial.suggest_categorical("penalty", all_penalties)

        # Filter penalties based on the solver
        if solver in ["liblinear", "saga"] and penalty not in ["l1", "l2"]:
            penalty = "l2"
        elif solver in ["newton-cg", "sag", "lbfgs"] and penalty != "l2":
            penalty = "l2"
        elif solver not in ["liblinear", "saga", "newton-cg", "sag", "lbfgs"]:
            penalty = None

        # 3️⃣ Wybór parametru C (dokładnie jak w param_grid)
        C = trial.suggest_categorical("C", [0.01, 0.1, 1, 10, 100])

        # 4️⃣ Tworzenie modelu z dokładnie tymi samymi parametrami co w param_grid
        model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=500)

        # 5️⃣ Walidacja krzyżowa (5-fold)
        score = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
        return score
    
    def visualize_results(self, study):
        plot_optimization_results(study, title="Logistic Regression Optimization")
        plot_best_parameters(study, title="Logistic Regression Best Parameters")

