"""
src/models.py
-------------
Model definitions for EEG Motor Imagery classification.

  1. Classical sklearn models (LR, SVM, RF)
  2. EEGNet — compact CNN for EEG (Lawhern et al., 2018)
  3. EEGNetClassifier — sklearn-compatible wrapper around EEGNet
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ── 1. Classical sklearn models ───────────────────────────────────────────────

def make_lr() -> Pipeline:
    """Logistic Regression with L2 regularisation + standard scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs",
            max_iter=1000, random_state=42
        )),
    ])


def make_svm() -> Pipeline:
    """SVM with RBF kernel — strong baseline for EEG classification."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf", C=10.0, gamma="scale",
            probability=True, random_state=42
        )),
    ])


def make_rf() -> Pipeline:
    """Random Forest — robust to feature scale, built-in importance."""
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=None,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )),
    ])


# ── 2. EEGNet architecture ────────────────────────────────────────────────────

class EEGNet(nn.Module):
    """
    EEGNet: A compact CNN for EEG-based BCIs.

    Reference:
        Lawhern VJ et al., "EEGNet: a compact convolutional neural network
        for EEG-based brain–computer interfaces",
        J. Neural Eng., 2018. DOI: 10.1088/1741-2552/aace8c

    Architecture (defaults for 64-ch, 160 Hz, 3s epochs):
        TemporalConv  : (1, 1, 64)  — temporal filter across all channels
        DepthwiseConv : (D, C, 1)   — per-channel spatial filter
        SeparableConv : (D*F1, 1, 16) — learns temporal summary features
        Classifier    : Dense(n_classes)

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (default: 64)
    n_times : int
        Number of time samples per epoch (default: 481 = 3s @ 160 Hz)
    n_classes : int
        Number of output classes (default: 2)
    F1 : int
        Number of temporal filters (default: 8)
    D : int
        Depth multiplier for depthwise conv (default: 2)
    F2 : int
        Number of pointwise filters (default: 16)
    dropout_rate : float
        Dropout probability (default: 0.5)
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 481,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times

        # Block 1: Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
        )

        # Block 2: Depthwise spatial convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate),
        )

        # Block 3: Separable convolution
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate),
        )

        # Compute flattened size dynamically
        self._flat_size = self._get_flat_size()

        # Classifier head
        self.classifier = nn.Linear(self._flat_size, n_classes)

    def _get_flat_size(self) -> int:
        """Compute flattened size by passing a dummy tensor."""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_times)
            x = self.temporal_conv(x)
            x = self.depthwise_conv(x)
            x = self.separable_conv(x)
            return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, 1, n_channels, n_times)
        """
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── 3. sklearn-compatible EEGNet wrapper ─────────────────────────────────────

class EEGNetClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for EEGNet.

    Expects X of shape (n_epochs, n_channels, n_times).
    Automatically normalises per-epoch and adds channel dimension.
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 481,
        n_classes: int = 2,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str | None = None,
        random_state: int = 42,
    ):
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state

    def _prepare_X(self, X: np.ndarray) -> torch.Tensor:
        """Normalise + add channel dim: (N, C, T) → (N, 1, C, T)."""
        # Per-epoch z-score
        mean = X.mean(axis=(1, 2), keepdims=True)
        std = X.std(axis=(1, 2), keepdims=True) + 1e-8
        X = (X - mean) / std
        return torch.FloatTensor(X[:, None, :, :])

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        n_channels = X.shape[1]
        n_times = X.shape[2]

        self.model_ = EEGNet(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=self.n_classes,
        ).to(self.device)

        X_t = self._prepare_X(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        criterion = nn.CrossEntropyLoss()

        self.train_losses_ = []
        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.model_(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            self.train_losses_.append(epoch_loss / len(loader))

        self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        X_t = self._prepare_X(X).to(self.device)
        with torch.no_grad():
            logits = self.model_(X_t)
            preds = logits.argmax(dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        X_t = self._prepare_X(X).to(self.device)
        with torch.no_grad():
            logits = self.model_(X_t)
            proba = F.softmax(logits, dim=1).cpu().numpy()
        return proba

    def save(self, path: str):
        torch.save(self.model_.state_dict(), path)

    def load(self, path: str):
        self.model_.load_state_dict(torch.load(path, map_location=self.device))
        return self


# ── Model registry ────────────────────────────────────────────────────────────

def get_all_models(n_channels: int = 64, n_times: int = 481) -> dict:
    """Return all models for comparison."""
    return {
        "LogisticRegression": make_lr(),
        "SVM_RBF":            make_svm(),
        "RandomForest":       make_rf(),
        "EEGNet":             EEGNetClassifier(
            n_channels=n_channels,
            n_times=n_times,
            epochs=80,
            batch_size=32,
            lr=1e-3,
        ),
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing EEGNet architecture …")
    net = EEGNet(n_channels=64, n_times=481)
    x = torch.zeros(8, 1, 64, 481)
    out = net(x)
    print(f"  Input : {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape)}")

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    print("\nTesting EEGNetClassifier …")
    rng = np.random.default_rng(42)
    X = rng.standard_normal((40, 64, 481)).astype(np.float32)
    y = np.array([0] * 20 + [1] * 20)
    clf = EEGNetClassifier(n_channels=64, n_times=481, epochs=5)
    clf.fit(X, y)
    preds = clf.predict(X)
    print(f"  Prediction shape: {preds.shape}")
    print(f"  Train acc: {(preds == y).mean():.2%}")
    print("\n✅ Model tests passed")
