from sklearn.model_selection import cross_val_score


class Cross_Validation:
    def __init__(self, n_splits=5, rand_st=42):
        self.K = n_splits
        self.rand_st = rand_st

        self.models = []
        self.mod_scores = []

        self.X_train, self.y_train = None
        self.X_test, self.y_test = None

        self.metric_names = ["Accuracy, Precision", "Recall", "F1"]

    def add_model(self, model):
        self.models.append(model)

    def split_data(self, split_fun):
        self.X_train, self.X_test, self.y_train, self.y_train = split_fun()

    def opt_hyperparams(self, hyperparams):
        pass

    def evaluate(self):
        y = self.y_train
        for model in self.models:
            y_pred = model.predict(self.y_test)
            pass

    # Model needs to have a set name
    def print_scores(self):
        for model, scores in zip(self.models, self.mod_scores):
            # Here
            print(f"Scores for {model.name}")
            for s in range(len(scores)):
                print(f"{self.metric_names[s]}: {scores[s]}")
