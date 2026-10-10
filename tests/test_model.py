import numpy as np

from src.model import NeuralNetwork


def test_softmax_rows_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0], [0.1, -0.2, 0.3]])
    probs = NeuralNetwork.softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0)


def test_prediction_shape_and_range():
    model = NeuralNetwork(seed=1)
    X = np.zeros((5, 64))
    probs = model.predict_proba(X)
    preds = model.predict(X)
    assert probs.shape == (5, 10)
    assert preds.shape == (5,)
    assert np.all((preds >= 0) & (preds <= 9))


def test_gradients_are_finite():
    model = NeuralNetwork(seed=2)
    X = np.random.default_rng(3).random((4, 64))
    y = np.array([0, 1, 2, 3])
    loss, grads = model.gradients(X, y)
    assert np.isfinite(loss)
    assert all(np.all(np.isfinite(g)) for g in grads.values())


def test_training_step_does_not_increase_batch_loss_too_much():
    model = NeuralNetwork(seed=4)
    X = np.random.default_rng(5).random((8, 64))
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    before, _ = model.gradients(X, y)
    model.train_batch(X, y, learning_rate=0.01)
    after, _ = model.gradients(X, y)
    assert after <= before + 1e-8
