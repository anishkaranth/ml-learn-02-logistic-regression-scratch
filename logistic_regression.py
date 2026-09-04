"""
Week 2 — Logistic Regression from Scratch (NumPy)

Educational script: synthetic 2D binary classification, sigmoid, BCE loss,
batch gradient descent, accuracy, and a decision-boundary plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 1. Synthetic 2D binary classification data
# ---------------------------------------------------------------------------

def make_blobs(
    n_per_class: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two Gaussian blobs in 2D — class 0 around (-1.5, -1.0), class 1 around (1.5, 1.0).
    Returns X of shape (2n, 2) and y of shape (2n,) with labels in {0, 1}.
    """
    rng = np.random.default_rng(seed)
    mean0 = np.array([-1.5, -1.0])
    mean1 = np.array([1.5, 1.0])
    cov = np.array([[0.6, 0.1], [0.1, 0.6]])

    x0 = rng.multivariate_normal(mean0, cov, size=n_per_class)
    x1 = rng.multivariate_normal(mean1, cov, size=n_per_class)
    X = np.vstack([x0, x1])
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])

    # Shuffle so batches / prints aren't all one class then the other
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# 2. Sigmoid and binary cross-entropy
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid: σ(z) = 1 / (1 + e^{-z}).

    Clip z so exp(-z) does not overflow for large |z|.
    """
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """
    Mean binary cross-entropy:

      L = -(1/n) * sum[ y * log(p) + (1 - y) * log(1 - p) ]

    Clamp probabilities away from 0/1 to avoid log(0).
    """
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


# ---------------------------------------------------------------------------
# 3. LogisticRegression class (from scratch)
# ---------------------------------------------------------------------------

class LogisticRegression:
    """
    Binary logistic regression trained with batch gradient descent.

    Model:  p(y=1 | x) = σ(w · x + b)
    Loss:   binary cross-entropy
    """

    def __init__(self, lr: float = 0.1, epochs: int = 300) -> None:
        self.lr = lr
        self.epochs = epochs
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        """Batch GD. X: (n, d), y: (n,) with labels in {0, 1}."""
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            logits = X @ self.w + self.b
            probs = sigmoid(logits)

            # Gradients of mean BCE w.r.t. w and b:
            #   dL/dw = (1/n) * X^T (p - y)
            #   dL/db = (1/n) * sum(p - y)
            err = probs - y
            dw = (X.T @ err) / n
            db = float(np.mean(err))

            self.w -= self.lr * dw
            self.b -= self.lr * db
            self.loss_history.append(binary_cross_entropy(y, probs))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(y=1 | x) for each row of X."""
        if self.w is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Hard labels: 1 if probability >= threshold, else 0."""
        return (self.predict_proba(X) >= threshold).astype(float)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


# ---------------------------------------------------------------------------
# 4. Decision-boundary plot
# ---------------------------------------------------------------------------

def plot_decision_boundary(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    out_path: Path,
) -> None:
    """Scatter of points + filled decision regions + loss curve."""
    assert model.w is not None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Left: scatter + decision regions ---
    pad = 1.0
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = model.predict_proba(grid).reshape(xx.shape)

    axes[0].contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.55)
    axes[0].contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=2)
    axes[0].scatter(
        X[y == 0, 0], X[y == 0, 1],
        c="C0", edgecolors="k", linewidths=0.4, label="class 0", alpha=0.85,
    )
    axes[0].scatter(
        X[y == 1, 0], X[y == 1, 1],
        c="C3", edgecolors="k", linewidths=0.4, label="class 1", alpha=0.85,
    )
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_title("Decision boundary (p = 0.5 contour)")
    axes[0].legend(loc="upper left")

    # --- Right: training loss ---
    axes[1].plot(model.loss_history, color="C0")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("BCE loss")
    axes[1].set_title("Training loss (gradient descent)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main() -> None:
    X, y = make_blobs(n_per_class=100, seed=42)

    model = LogisticRegression(lr=0.5, epochs=400)
    model.fit(X, y)

    probs = model.predict_proba(X)
    preds = model.predict(X)
    acc = accuracy(y, preds)

    print("=== Logistic Regression from Scratch ===")
    print(f"Samples: {len(y)}  |  Features: {X.shape[1]}")
    print(f"Weights w: {np.round(model.w, 4)}  |  bias b: {model.b:.4f}")
    print(f"Final BCE loss: {model.loss_history[-1]:.4f}")
    print(f"Train accuracy: {acc * 100:.1f}%")
    print(f"Mean predicted P(y=1): {probs.mean():.3f}")

    out = Path(__file__).resolve().parent / "outputs" / "decision_boundary.png"
    plot_decision_boundary(model, X, y, out)
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
