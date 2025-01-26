from model import Model_clf
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

log = Model_clf("Logistic Regression")

param_grid = {
    "C": [0.001, 0.1, 1, 10, 100],
    "penality": ["l1", "l2"],
    "solver": ["saga", "liblinear"],
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000), param_grid, cv=5, scoring={""}
)


class Logistic_Regression:
    def __init__(self, name):
        self.name = name
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def set_params(self, **params):
        param_grid = {
            "C": [0.001, 0.1, 1, 10, 100],
            "penality": ["l1", "l2"],
            "solver": ["saga", "liblinear"],
        }

    def data_preprocessing(self, data):
        pass

    def gridsearch(X_train, y_train):
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_
