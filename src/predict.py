"""Predict a digit from a test-set example."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from src.model import NeuralNetwork

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="test example index")
    args = parser.parse_args()

    model_path = ROOT / "artifacts" / "model.npz"
    if not model_path.exists():
        raise SystemExit("Model not found. Run `python -m src.train` first.")

    digits = load_digits()
    X = digits.data.astype(float) / 16.0
    y = digits.target
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if not 0 <= args.index < len(X_test):
        raise SystemExit(f"Index must be between 0 and {len(X_test) - 1}.")

    model = NeuralNetwork.load(model_path)
    image = X_test[args.index].reshape(8, 8)
    prediction = int(model.predict(X_test[args.index:args.index + 1])[0])
    actual = int(y_test[args.index])
    print(f"Predicted digit: {prediction}")
    print(f"Actual digit:    {actual}")
    print(f"Correct:         {prediction == actual}")

    plt.imshow(image, cmap="gray_r", interpolation="nearest")
    plt.title(f"Predicted: {prediction} | Actual: {actual}")
    plt.axis("off")
    plt.tight_layout()
    output = ROOT / "artifacts" / f"prediction_{args.index}.png"
    plt.savefig(output, dpi=160)
    print(f"Image saved to: {output}")


if __name__ == "__main__":
    main()
