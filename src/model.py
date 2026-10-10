"""A small fully connected neural network implemented with NumPy only."""
from __future__ import annotations

import numpy as np


class NeuralNetwork:
    """64 → hidden → 10 classifier using ReLU and softmax."""

    def __init__(self, input_size=64, hidden_size=32, output_size=10, seed=42):
        rng = np.random.default_rng(seed)
        # He initialization is suitable for layers followed by ReLU.
        self.W1 = rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(logits):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    @staticmethod
    def one_hot(y, num_classes=10):
        return np.eye(num_classes)[np.asarray(y, dtype=int)]

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        logits = a1 @ self.W2 + self.b2
        probs = self.softmax(logits)
        cache = (X, z1, a1, probs)
        return probs, cache

    @staticmethod
    def cross_entropy(probs, y):
        y = np.asarray(y, dtype=int)
        eps = 1e-12
        return float(-np.mean(np.log(probs[np.arange(len(y)), y] + eps)))

    def gradients(self, X, y):
        """Return loss and gradients for a batch."""
        y = np.asarray(y, dtype=int)
        probs, cache = self.forward(X)
        X, z1, a1, probs = cache
        m = len(X)
        y_hot = self.one_hot(y, self.b2.shape[1])

        dz2 = (probs - y_hot) / m
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        return self.cross_entropy(probs, y), {
            "W1": dW1, "b1": db1, "W2": dW2, "b2": db2
        }

    def train_batch(self, X, y, learning_rate=0.1):
        loss, grads = self.gradients(X, y)
        for name, grad in grads.items():
            setattr(self, name, getattr(self, name) - learning_rate * grad)
        return loss

    def predict_proba(self, X):
        return self.forward(np.asarray(X, dtype=float))[0]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def save(self, path):
        np.savez(
            path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
            input_size=self.W1.shape[0], hidden_size=self.W1.shape[1],
            output_size=self.W2.shape[1],
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(
            int(data["input_size"]), int(data["hidden_size"]),
            int(data["output_size"])
        )
        model.W1, model.b1 = data["W1"], data["b1"]
        model.W2, model.b2 = data["W2"], data["b2"]
        return model
