from sklearn.ensemble import RandomForestClassifier
from models.model_ import Model_
from sklearn.model_selection import cross_val_score


class Random_Forest_(Model_):
    def __init__(self):
        self.model = RandomForestClassifier()
        super().__init__(self.model)

    def objective(self, trial, X, y):
        return super().objective(trial, X, y)
