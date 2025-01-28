from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics import make_scorer, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from models.model_ import Model_
from visualization import plot_optimization_results, plot_best_parameters


class Random_Forest_(Model_):
    def __init__(self):
        self.model = RandomForestClassifier()
        super().__init__(self.model)

    def objective(self, trial, X, y):
        # Define hyperparameter search space
        n_estimators = trial.suggest_int("n_estimators", 10, 500)
        max_depth = trial.suggest_int("max_depth", 1, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

        # Create the model with the sampled hyperparameters
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
        )

        # Use recall as the scoring metric
        recall_scorer = make_scorer(recall_score, average="weighted")
        scores = cross_val_score(model, X, y, cv=5, scoring=recall_scorer)

        return scores.mean()

    def visualize_results(self, study):
        # Wizualizacja wyników optymalizacji
        plot_optimization_results(study, title="Random Forest Optimization")
        # Wizualizacja najlepszych parametrów
        plot_best_parameters(study, title="Random Forest Best Parameters")
