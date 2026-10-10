"""Train and evaluate the NumPy neural network."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

from src.model import NeuralNetwork

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = NeuralNetwork(input_size=64, hidden_size=32, output_size=10, seed=42)
    epochs = 100
    batch_size = 32
    learning_rate = 0.08
    rng = np.random.default_rng(42)
    train_losses, train_accuracies, test_accuracies = [], [], []

    for epoch in range(epochs):
        order = rng.permutation(len(X_train))
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            epoch_losses.append(
                model.train_batch(X_train[idx], y_train[idx], learning_rate)
            )

        train_loss = float(np.mean(epoch_losses))
        train_acc = model.accuracy(X_train, y_train)
        test_acc = model.accuracy(X_test, y_test)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:03d}/{epochs} | "
                f"loss={train_loss:.4f} | "
                f"train_acc={train_acc:.4f} | test_acc={test_acc:.4f}"
            )

    predictions = model.predict(X_test)
    final_metrics = {
        "dataset": "scikit-learn digits",
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_accuracy": model.accuracy(X_train, y_train),
        "test_accuracy": model.accuracy(X_test, y_test),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }

    model.save(ARTIFACTS / "model.npz")
    with (ARTIFACTS / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, epochs + 1), train_losses, label="Training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "training_loss.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(1, epochs + 1), train_accuracies, label="Train accuracy")
    ax.plot(range(1, epochs + 1), test_accuracies, label="Test accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "accuracy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, ax=ax, colorbar=False
    )
    ax.set_title("Test-set confusion matrix")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print("\nFinal results")
    print(f"Train accuracy: {final_metrics['train_accuracy']:.2%}")
    print(f"Test accuracy:  {final_metrics['test_accuracy']:.2%}")
    print(f"Artifacts saved to: {ARTIFACTS}")


if __name__ == "__main__":
    main()
