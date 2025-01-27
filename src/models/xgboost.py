from models.model_ import Model_
import xgboost as xgb


class XGBoost(Model_):
    def __init__(self):
        self.model = xgb.XGBClassifier()
        super().__init__(self.model)

    def objective(self, trial, X, y):
        return super().objective(trial, X, y)
