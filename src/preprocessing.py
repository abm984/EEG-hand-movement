"""
src/preprocessing.py
--------------------
MNE-Python preprocessing pipeline for OpenNeuro ds004362.

Steps:
  1. Load EDF files (BIDS or PhysioNet format)
  2. Bandpass filter 1–40 Hz (Butterworth, order 5)
  3. Notch filter 50 Hz (power-line)
  4. ICA: remove eye-blink + muscle artifacts
  5. Epoch around T1/T2 events [-0.5, 2.5 s]
  6. Baseline correction [-0.5, 0 s]
  7. Epoch rejection: peak-to-peak > 100 µV
  8. Return (epochs_left, epochs_right, info)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import mne
import numpy as np

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SFREQ = 160.0          # Hz — original sampling rate in ds004362
L_FREQ = 1.0           # Hz — high-pass cutoff
H_FREQ = 40.0          # Hz — low-pass cutoff
NOTCH_FREQ = 50.0      # Hz — power-line
EPOCH_TMIN = -0.5      # s  — epoch start relative to event
EPOCH_TMAX = 2.5       # s  — epoch end
BASELINE = (-0.5, 0.0) # s  — baseline window
REJECT_THRESH = 300e-6 # V  — peak-to-peak rejection threshold (300 uV)
N_ICA_COMPONENTS = 20

# Imagery runs: T1 = left fist, T2 = right fist
IMAGERY_RUNS = [4, 8, 12]
EVENT_ID = {"left_fist": 1, "right_fist": 2}  # T1=1, T2=2


# ── Data loading ──────────────────────────────────────────────────────────────

def load_subject_raw(
    subject_id: str,
    data_dir: str | Path,
    runs: list[int] = IMAGERY_RUNS,
    verbose: bool = False,
) -> mne.io.Raw:
    """
    Load and concatenate EDF runs for one subject.

    Supports both BIDS layout (sub-XXX/eeg/*.edf) and
    PhysioNet flat layout (SXXX/SXXXRXX.edf).

    Parameters
    ----------
    subject_id : str
        e.g. "sub-001" (BIDS) or "S001" (PhysioNet)
    data_dir : Path
        Root of the downloaded dataset
    runs : list[int]
        Run numbers to load (default: [4, 8, 12] for imagery)

    Returns
    -------
    mne.io.Raw
        Concatenated raw recording
    """
    data_dir = Path(data_dir)
    raws = []

    for run_num in runs:
        # Try BIDS layout first (checking both .edf and .set, and both single/double digit runs)
        files = []
        for ext in [".edf", ".set"]:
            for r_str in [f"{run_num}", f"{run_num:02d}"]:
                bids_pattern = f"{subject_id}/eeg/{subject_id}_task-*_run-{r_str}_eeg{ext}"
                files = list(data_dir.glob(bids_pattern))
                if files:
                    break
            if files:
                break

        if not files:
            # Fallback: PhysioNet layout (checking both .edf and .set, and both single/double digit runs)
            phys_sub = subject_id.replace("sub-0", "S").replace("sub-", "S")
            for ext in [".edf", ".set"]:
                for r_str in [f"{run_num}", f"{run_num:02d}"]:
                    phys_pattern = f"{phys_sub}/{phys_sub}R{r_str}{ext}"
                    files = list(data_dir.glob(phys_pattern))
                    if files:
                        break
                if files:
                    break

        if not files:
            logger.warning(f"Run {run_num} not found for {subject_id}, skipping")
            continue

        # Use general read_raw, which supports both formats automatically
        raw = mne.io.read_raw(files[0], preload=True, verbose=verbose)
        raws.append(raw)

    if not raws:
        raise FileNotFoundError(
            f"No EDF or SET files found for {subject_id} in {data_dir}. "
            "Run src/download.py first."
        )

    return mne.concatenate_raws(raws)


# ── Preprocessing steps ───────────────────────────────────────────────────────

def apply_filters(raw: mne.io.Raw) -> mne.io.Raw:
    """Bandpass 1–40 Hz + notch 50 Hz."""
    raw = raw.copy()
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, method="iir",
               iir_params=dict(order=5, ftype="butter"), verbose=False)
    raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)
    return raw


def run_ica(raw: mne.io.Raw, n_components: int = N_ICA_COMPONENTS) -> mne.io.Raw:
    """
    Run FastICA and auto-reject EOG/EMG components.

    Uses MNE's built-in correlation with EOG/ECG pseudo-channels.
    Falls back gracefully if no frontal channels are found.
    """
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="fastica",
        random_state=42,
        max_iter=500,
        verbose=False,
    )
    ica.fit(raw, verbose=False)

    # Auto-detect eye-blink components via frontal channel correlation
    exclude_idx = []
    frontal_chs = [ch for ch in raw.ch_names if ch.startswith(("Fp", "AF"))]
    if frontal_chs:
        eog_idx, _ = ica.find_bads_eog(raw, ch_name=frontal_chs[0], verbose=False)
        exclude_idx.extend(eog_idx)

    ica.exclude = list(set(exclude_idx))[:3]  # exclude at most 3 components
    raw_clean = ica.apply(raw.copy(), verbose=False)
    logger.debug(f"ICA excluded components: {ica.exclude}")
    return raw_clean


def extract_events(raw: mne.io.Raw) -> np.ndarray:
    """
    Extract T1/T2 events from the raw annotations.

    In BCI2000/PhysioNet EDF files, events are stored as annotations
    with descriptions 'T1' and 'T2'. In the BIDS version, they may be
    stored with names like 'TASK2T1'/'TASK2T2'. We map both formats.
    """
    unique_descs = set(raw.annotations.description)
    logger.debug(f"Available annotations in raw: {unique_descs}")

    # Build a flexible event mapping based on present annotations
    event_id = {}
    for desc in unique_descs:
        if desc in ["T1", "TASK2T1", "TASK1T1"]:
            event_id[desc] = 1
        elif desc in ["T2", "TASK2T2", "TASK1T2"]:
            event_id[desc] = 2

    # Fallback to numeric values if strings are not matched
    if not event_id:
        for desc in unique_descs:
            if desc in ["1", 1]:
                event_id[desc] = 1
            elif desc in ["2", 2]:
                event_id[desc] = 2

    if not event_id:
        raise ValueError(
            f"No recognizable motor imagery annotations found (T1/T2/TASK2T1/TASK2T2). "
            f"Found annotations: {unique_descs}"
        )

    events, _ = mne.events_from_annotations(
        raw,
        event_id=event_id,
        verbose=False,
    )
    return events


def make_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    tmin: float = EPOCH_TMIN,
    tmax: float = EPOCH_TMAX,
    baseline: tuple = BASELINE,
    reject_thresh: float = REJECT_THRESH,
) -> mne.Epochs:
    """Create baseline-corrected epochs, reject noisy ones."""
    reject = {"eeg": reject_thresh}
    epochs = mne.Epochs(
        raw,
        events,
        event_id=EVENT_ID,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        reject=reject,
        preload=True,
        verbose=False,
    )
    n_before = len(epochs)
    epochs.drop_bad(verbose=False)
    n_after = len(epochs)
    logger.debug(f"Rejected {n_before - n_after}/{n_before} epochs")
    return epochs


# ── Full pipeline ─────────────────────────────────────────────────────────────

def preprocess_subject(
    subject_id: str,
    data_dir: str | Path,
    use_ica: bool = True,
    runs: list[int] = IMAGERY_RUNS,
    verbose: bool = False,
) -> tuple[mne.Epochs, dict]:
    """
    Full preprocessing pipeline for one subject.

    Returns
    -------
    epochs : mne.Epochs
        Cleaned epochs with event_id {'left_fist': 1, 'right_fist': 2}
    info_dict : dict
        Metadata: subject_id, n_epochs, n_dropped, sfreq, ch_names
    """
    logger.info(f"Processing {subject_id} …")

    # 1. Load
    raw = load_subject_raw(subject_id, data_dir, runs=runs, verbose=verbose)
    n_channels = len(raw.ch_names)
    sfreq = raw.info["sfreq"]

    # 2. Filter
    raw = apply_filters(raw)

    # 3. ICA artifact removal
    if use_ica:
        raw = run_ica(raw)

    # 4. Extract events
    events = extract_events(raw)

    # 5. Epoch + reject
    epochs = make_epochs(raw, events)

    info_dict = {
        "subject_id": subject_id,
        "n_channels": n_channels,
        "sfreq": sfreq,
        "n_epochs_total": sum(events[:, 2] != 0),
        "n_epochs_kept": len(epochs),
        "n_left": len(epochs["left_fist"]),
        "n_right": len(epochs["right_fist"]),
    }
    logger.info(
        f"  {subject_id}: {info_dict['n_left']} left, "
        f"{info_dict['n_right']} right epochs kept"
    )
    return epochs, info_dict


def preprocess_cohort(
    subject_ids: list[str],
    data_dir: str | Path,
    use_ica: bool = True,
    verbose: bool = False,
) -> tuple[list[mne.Epochs], list[dict]]:
    """
    Preprocess a list of subjects.

    Returns
    -------
    all_epochs : list of mne.Epochs
    all_info : list of dict
    """
    all_epochs, all_info = [], []
    for sub_id in subject_ids:
        try:
            epochs, info = preprocess_subject(
                sub_id, data_dir, use_ica=use_ica, verbose=verbose
            )
            all_epochs.append(epochs)
            all_info.append(info)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {sub_id}: {e}")
        except Exception as e:
            logger.error(f"Error processing {sub_id}: {e}")
    return all_epochs, all_info


# ── Convenience: get X, y arrays ─────────────────────────────────────────────

def epochs_to_array(
    epochs: mne.Epochs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert epochs to (X, y) arrays.

    Returns
    -------
    X : ndarray, shape (n_epochs, n_channels, n_times)
    y : ndarray, shape (n_epochs,), values 0=left / 1=right
    """
    X = epochs.get_data(units="uV")          # µV, shape (n, 64, times)
    y = (epochs.events[:, 2] == 2).astype(int)  # 0=left(T1), 1=right(T2)
    return X, y


def get_pooled_arrays(
    all_epochs: list[mne.Epochs],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pool all subjects into a single (X, y) array with subject labels.

    Returns
    -------
    X : ndarray (total_epochs, n_channels, n_times)
    y : ndarray (total_epochs,)
    subject_ids : ndarray (total_epochs,) — for LeaveOneGroupOut CV
    """
    Xs, ys, sids = [], [], []
    for i, epochs in enumerate(all_epochs):
        X, y = epochs_to_array(epochs)
        Xs.append(X)
        ys.append(y)
        sids.append(np.full(len(y), i))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(sids)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    subjects = ["sub-001", "sub-002"]

    print(f"Testing preprocessing on {subjects} from {data_dir}")
    all_epochs, all_info = preprocess_cohort(subjects, data_dir, verbose=False)

    for info in all_info:
        print(f"  {info['subject_id']}: left={info['n_left']}, right={info['n_right']}")

    X, y, sids = get_pooled_arrays(all_epochs)
    print(f"\nPooled arrays:")
    print(f"  X shape : {X.shape}")
    print(f"  y shape : {y.shape}  (0=left, 1=right)")
    print(f"  Class balance: {np.bincount(y)}")
