# Neural Network From Scratch

A small, educational handwritten-digit classifier built with **NumPy**. It implements a fully connected neural network without PyTorch or TensorFlow.

## Architecture

```text
8×8 grayscale image (64 features)
              ↓
      Dense layer: 64 → 32
              ↓
           ReLU
              ↓
      Dense layer: 32 → 10
              ↓
           Softmax
              ↓
       Predicted digit (0–9)
```

The project uses the scikit-learn `digits` dataset (1,797 samples; 8×8 images) as a convenient dataset loader. The model's forward pass, loss, gradients, backpropagation, and parameter updates are implemented with NumPy.

## Features

- NumPy implementation of dense layers, ReLU, softmax, cross-entropy loss, and backpropagation
- Train/test split to evaluate generalization on unseen examples
- Reproducible random seed
- Accuracy and loss tracking
- Confusion matrix and training-curve plots
- Save and load trained model parameters
- Predict a sample image from the test set

## Quick start

Python 3.10+ is recommended.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Train and evaluate
python -m src.train

# 4. Predict a test example
python -m src.predict --index 0
```

Training creates `artifacts/model.npz`, `artifacts/metrics.json`, and plots in `artifacts/`.

## Results

Run the project locally and record the actual test accuracy here. Results can vary with hyperparameters and environment.

| Metric | Result |
|---|---:|
| Test accuracy | Run `python -m src.train` |
| Dataset | scikit-learn digits |
| Train/test split | 80% / 20% |
| Hidden layer | 32 neurons, ReLU |
| Output layer | 10 classes, softmax |
| Optimizer | Mini-batch gradient descent |

**Important:** Training accuracy is not the same as test accuracy. Report the measured test score from your run rather than copying a score from a social post.

## Project structure

```text
neural-network-from-scratch/
├── artifacts/             # Generated model, metrics, and plots
├── src/
│   ├── __init__.py
│   ├── model.py            # NumPy neural network and backpropagation
│   ├── train.py            # Data loading, training, evaluation, plots
│   └── predict.py          # Load model and predict a test example
├── tests/
│   └── test_model.py       # Basic model and gradient checks
├── .gitignore
├── requirements.txt
└── README.md
```

## How backpropagation works

1. **Forward pass:** compute `Z1 = XW1 + b1`, apply ReLU, then compute output logits.
2. **Softmax:** turn logits into class probabilities.
3. **Cross-entropy:** measure the difference between predicted probabilities and true labels.
4. **Backward pass:** use the chain rule to calculate gradients for both layers.
5. **Update:** subtract the learning-rate-scaled gradients from the parameters.

For a batch of `m` examples, the output-layer gradient for softmax with cross-entropy is:

```text
dZ2 = (probabilities - one_hot_labels) / m
```

The ReLU derivative is 1 where its input is positive and 0 otherwise.

## Dataset and preprocessing

The `digits` dataset contains 1,797 labelled 8×8 grayscale images. Pixel intensities range from 0 to 16, so inputs are scaled to `[0, 1]`. The data is split with stratification to keep class proportions similar in the training and test sets.

## Limitations and next steps

- This is a small educational dataset, not a production OCR system.
- The model expects 8×8 flattened inputs; real photographs require preprocessing and resizing.
- The implementation uses a basic mini-batch gradient descent optimizer.
- Try changing hidden-layer size, learning rate, batch size, or activation function.
- Add a comparison with logistic regression, numerical gradient checks, or a small interactive demo.

## Tech stack

Python · NumPy · scikit-learn (dataset loading and splitting) · Matplotlib · pytest

## License

MIT — see `LICENSE` if you choose to add one before publishing.
