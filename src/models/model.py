from abc import ABC, abstractmethod


class Model_Clf(ABC):
    def __init__(self, model):
        self.name = model.__class__.__name__
        self.model = model
        self.best_params = None
        self.best_estimator = None

    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test, y_test):
        pass

    @abstractmethod
    def data_preprocessing(self, data):
        pass

    @abstractmethod
    def gridsearch(self, X_train, y_train, param_grid):
        pass
