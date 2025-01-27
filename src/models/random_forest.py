from sklearn.ensemble import RandomForestClassifier
from models.model_ import Model_
from sklearn.model_selection import cross_val_score


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
            random_state=42
        )

        # Perform cross-validation and return the mean score
        scores = cross_val_score(model, X, y, cv=3)

        return scores.mean()
