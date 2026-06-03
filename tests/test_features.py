"""
tests/test_features.py
-----------------------
Unit tests for feature extraction — no real EEG data needed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from features import BandPowerExtractor, CSPBandPower, make_csp_pipeline, make_psd_pipeline


RNG = np.random.default_rng(42)
N_EPOCHS, N_CH, N_TIMES = 40, 64, 481
X_FAKE = RNG.standard_normal((N_EPOCHS, N_CH, N_TIMES)).astype(np.float32)
Y_FAKE = np.array([0] * 20 + [1] * 20)


class TestBandPowerExtractor:
    def test_output_shape(self):
        bpe = BandPowerExtractor(sfreq=160.0)
        X_out = bpe.fit_transform(X_FAKE)
        n_bands = 5
        assert X_out.shape == (N_EPOCHS, N_CH * n_bands), \
            f"Expected {(N_EPOCHS, N_CH * n_bands)}, got {X_out.shape}"

    def test_all_finite(self):
        bpe = BandPowerExtractor(sfreq=160.0)
        X_out = bpe.fit_transform(X_FAKE)
        assert np.all(np.isfinite(X_out)), "BandPower output contains NaN/Inf"

    def test_single_epoch(self):
        bpe = BandPowerExtractor(sfreq=160.0)
        X_single = X_FAKE[:1]
        X_out = bpe.fit_transform(X_single)
        assert X_out.shape[0] == 1

    def test_custom_bands(self):
        custom_bands = {"alpha": (8, 13)}
        bpe = BandPowerExtractor(sfreq=160.0, bands=custom_bands)
        X_out = bpe.fit_transform(X_FAKE)
        assert X_out.shape == (N_EPOCHS, N_CH * 1)


class TestCSPBandPower:
    def test_output_shape(self):
        n_csp = 4
        csp = CSPBandPower(sfreq=160.0, n_components=n_csp)
        X_out = csp.fit_transform(X_FAKE, Y_FAKE)
        n_bands = 5
        assert X_out.shape == (N_EPOCHS, n_csp * n_bands), \
            f"Expected {(N_EPOCHS, n_csp * n_bands)}, got {X_out.shape}"

    def test_fit_then_transform(self):
        csp = CSPBandPower(sfreq=160.0, n_components=4)
        csp.fit(X_FAKE, Y_FAKE)
        X_out = csp.transform(X_FAKE)
        assert X_out.shape[0] == N_EPOCHS

    def test_all_finite(self):
        csp = CSPBandPower(sfreq=160.0, n_components=4)
        X_out = csp.fit_transform(X_FAKE, Y_FAKE)
        assert np.all(np.isfinite(X_out))


class TestPipelines:
    def test_csp_pipeline(self):
        pipe = make_csp_pipeline(sfreq=160.0, n_csp=4)
        X_out = pipe.fit_transform(X_FAKE, Y_FAKE)
        assert X_out.ndim == 2
        assert X_out.shape[0] == N_EPOCHS

    def test_psd_pipeline(self):
        pipe = make_psd_pipeline(sfreq=160.0, n_pca=20)
        X_out = pipe.fit_transform(X_FAKE)
        assert X_out.shape == (N_EPOCHS, 20)

    def test_pipeline_sklearn_compatible(self):
        """Pipeline must work inside sklearn's cross_validate."""
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([
            ("features", make_csp_pipeline(sfreq=160.0, n_csp=4)),
            ("clf", LogisticRegression(max_iter=200)),
        ])
        scores = cross_val_score(pipe, X_FAKE, Y_FAKE, cv=3, scoring="accuracy")
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)
