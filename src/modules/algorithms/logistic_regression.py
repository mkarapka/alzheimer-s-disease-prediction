import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learnign_rate, epochs, beta):
        self.lr = learnign_rate
        self.epochs = epochs
        self.β = beta

    def sigmoid(self, t):
        return np.where(t >= 0, 1 / (1 + np.exp(-t)), np.exp(t) / (1 + np.exp(t)))

    def log_likelihood(self, X, y):
        e = 1e-15
        Xβ = np.dot(X, self.β)
        s_x = self.sigmoid(Xβ)
        s_x = np.clip(s_x, e, 1 - e)
        return np.sum(y * np.log(s_x) + (1 - y) * np.log(1 - s_x))

    def fit_grad_asc(self, X, y):
        losses = []
        for epoch in range(self.epochs):
            Xβ = np.dot(X, self.β)
            y_pred = self.sigmoid(Xβ)

            gradient = np.dot(X.T, (y - y_pred))
            self.β += self.lr * gradient

            loss = -1 * self.log_likelihood(X, y)
            losses.append(loss)
        return self.β, losses


# test
X, y = make_classification(
    n_samples=100, n_features=30, n_informative=2, n_redundant=0, n_clusters_per_class=1
)

β_sample = np.random.random(X.shape[1])
log_reg = LogisticRegression(0.01, 1000, β_sample)

β, losses = log_reg.fit_grad_asc(X, y)
plt.plot(losses)
plt.title("Loss over time")
plt.xlabel("Epoch")
plt.ylabel("Loss")


plt.show()
