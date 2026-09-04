# Week 2 — Logistic Regression from Scratch (NumPy)

## Learning goal

Build intuition for **binary classification** by implementing **logistic regression** yourself with NumPy: the sigmoid, binary cross-entropy (BCE) loss, gradient descent, and a decision-boundary visualization on a synthetic 2D dataset.

## What you'll build

- Synthetic 2D binary classification data (two Gaussian blobs — no downloads)
- Sigmoid activation that maps scores to probabilities in \((0, 1)\)
- Binary cross-entropy loss and its gradients
- A `LogisticRegression` class with `fit` (batch GD), `predict_proba`, and `predict`
- Plots: scatter + decision boundary, and loss over epochs

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Script (VS Code / terminal):**

```bash
python logistic_regression.py
```

This trains the model, prints loss / accuracy, and saves `outputs/decision_boundary.png`.

**Notebook (interactive):**

```bash
jupyter notebook notebooks/logistic_regression_scratch.ipynb
```

Both share the same core ideas. The notebook adds sigmoid visualization, markdown intuition, and inline plots; the `.py` script is a clean, runnable walkthrough.

## Requirements

- Python 3.9+
- `numpy`, `matplotlib`, `jupyter` (see `requirements.txt`)
- No scikit-learn — the ML core and toy data are from scratch

## What you'll learn

- Why we use a sigmoid (not a straight line) for classification probabilities
- How BCE loss penalizes confident wrong predictions
- How gradient descent updates weights and bias for logistic regression
- How a linear decision boundary separates two classes in 2D feature space
