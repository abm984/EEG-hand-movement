"""
src/features.py
---------------
Feature extraction for EEG motor imagery classification.

Three feature sets:
  A. CSP band power   — 6 CSP components × 5 bands → 30 features
  B. PSD channel mean — 5 bands × 64 channels → 320 features (→ PCA)
  C. Raw epochs       — used directly by EEGNet CNN (no hand-crafted features)
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from mne.decoding import CSP

logger = logging.getLogger(__name__)

# ── Frequency band definitions ────────────────────────────────────────────────
BANDS = {
    "delta":  (1.0,  4.0),
    "theta":  (4.0,  8.0),
    "alpha":  (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}


# ── Band-power extractor (Welch PSD per channel) ──────────────────────────────

class BandPowerExtractor(BaseEstimator, TransformerMixin):
    """
    Compute mean power in each frequency band per channel.

    Input  : X shape (n_epochs, n_channels, n_times)
    Output : X shape (n_epochs, n_channels × n_bands)
             e.g. 64 channels × 5 bands = 320 features
    """

    def __init__(self, sfreq: float = 160.0, bands: dict = None):
        self.sfreq = sfreq
        self.bands = bands or BANDS

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        n_epochs, n_channels, n_times = X.shape
        n_bands = len(self.bands)
        features = np.zeros((n_epochs, n_channels * n_bands))

        nperseg = min(int(self.sfreq), n_times)

        for i in range(n_epochs):
            for c in range(n_channels):
                freqs, psd = welch(X[i, c, :], fs=self.sfreq, nperseg=nperseg)
                for b_idx, (lo, hi) in enumerate(self.bands.values()):
                    mask = (freqs >= lo) & (freqs <= hi)
                    power = np.mean(psd[mask]) if mask.sum() > 0 else 0.0
                    # log-transform for more Gaussian-like distribution
                    features[i, c * n_bands + b_idx] = np.log1p(power)

        return features


# ── Log-variance of CSP-filtered epochs (gold standard for MI) ───────────────

class CSPBandPower(BaseEstimator, TransformerMixin):
    """
    For each frequency band:
      1. Bandpass-filter epochs into the band
      2. Apply CSP spatial filter
      3. Compute log-variance of each CSP component

    Input  : X shape (n_epochs, n_channels, n_times)
    Output : X shape (n_epochs, n_csp_components × n_bands)
    """

    def __init__(
        self,
        sfreq: float = 160.0,
        n_components: int = 6,
        bands: dict = None,
    ):
        self.sfreq = sfreq
        self.n_components = n_components
        self.bands = bands or BANDS
        self.csp_filters_: dict[str, CSP] = {}

    def _bandpass(self, X: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Simple IIR bandpass via scipy."""
        from scipy.signal import butter, sosfiltfilt
        nyq = self.sfreq / 2.0
        low = max(lo / nyq, 0.001)
        high = min(hi / nyq, 0.999)
        sos = butter(5, [low, high], btype="band", output="sos")
        return sosfiltfilt(sos, X, axis=-1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        for band_name, (lo, hi) in self.bands.items():
            X_band = self._bandpass(X, lo, hi)
            csp = CSP(n_components=self.n_components, reg=None,
                      log=True, norm_trace=False)
            csp.fit(X_band, y)
            self.csp_filters_[band_name] = csp
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        features_per_band = []
        for band_name, (lo, hi) in self.bands.items():
            X_band = self._bandpass(X, lo, hi)
            csp = self.csp_filters_[band_name]
            feat = csp.transform(X_band)  # shape (n_epochs, n_components)
            features_per_band.append(feat)
        return np.concatenate(features_per_band, axis=1)


# ── Normaliser for CSP output ─────────────────────────────────────────────────

class EpochScaler(BaseEstimator, TransformerMixin):
    """Subtract per-epoch mean and divide by per-epoch std."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:  # (n_epochs, n_channels, n_times)
            mean = X.mean(axis=-1, keepdims=True)
            std = X.std(axis=-1, keepdims=True) + 1e-8
            return (X - mean) / std
        return X  # already 2D features


# ── Feature pipeline builders ─────────────────────────────────────────────────

def make_csp_pipeline(
    sfreq: float = 160.0,
    n_csp: int = 6,
    n_pca: int | None = None,
) -> Pipeline:
    """
    CSP + log-variance pipeline for classical ML models.

    Returns sklearn Pipeline that transforms
    (n_epochs, n_channels, n_times) → (n_epochs, n_features).
    """
    steps = [("csp_band_power", CSPBandPower(sfreq=sfreq, n_components=n_csp))]
    if n_pca:
        steps.append(("pca", PCA(n_components=n_pca, random_state=42)))
    return Pipeline(steps)


def make_psd_pipeline(
    sfreq: float = 160.0,
    n_pca: int = 50,
) -> Pipeline:
    """PSD band-mean features + PCA for dimensionality reduction."""
    return Pipeline([
        ("band_power", BandPowerExtractor(sfreq=sfreq)),
        ("pca", PCA(n_components=n_pca, random_state=42)),
    ])


# ── Summary statistics ────────────────────────────────────────────────────────

def summarise_features(X_feat: np.ndarray, feature_type: str = ""):
    """Print a brief feature summary."""
    print(f"\nFeature summary {feature_type}:")
    print(f"  Shape  : {X_feat.shape}")
    print(f"  Mean   : {X_feat.mean():.4f}")
    print(f"  Std    : {X_feat.std():.4f}")
    print(f"  Min/Max: {X_feat.min():.4f} / {X_feat.max():.4f}")
    finite = np.all(np.isfinite(X_feat))
    print(f"  Finite : {finite}")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # Fake data: 40 epochs, 64 channels, 3 seconds at 160 Hz
    X_fake = rng.standard_normal((40, 64, 481)).astype(np.float32)
    y_fake = np.array([0] * 20 + [1] * 20)

    print("Testing BandPowerExtractor …")
    bpe = BandPowerExtractor(sfreq=160.0)
    X_bpe = bpe.fit_transform(X_fake)
    summarise_features(X_bpe, "BandPower")

    print("\nTesting CSPBandPower …")
    csp_bp = CSPBandPower(sfreq=160.0, n_components=4)
    X_csp = csp_bp.fit_transform(X_fake, y_fake)
    summarise_features(X_csp, "CSP")

    print("\nTesting full CSP pipeline …")
    pipe = make_csp_pipeline(sfreq=160.0, n_csp=4)
    X_pipe = pipe.fit_transform(X_fake, y_fake)
    summarise_features(X_pipe, "CSP pipeline")

    print("\n✅ All feature tests passed")
