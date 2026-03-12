import optuna


class Model_:
    def __init__(self, model):
        self.name = model.__class__.__name__
        self.model = model
        self.best_params = None
        self.best_estimator = None

    def bayesian_opt(self, X, y, n_trials=100):
        model = type(self.model)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: self.objective(t, X, y), n_trials=n_trials)

        self.best_params = study.best_params
        self.best_estimator = model(**self.best_params)

        return study

