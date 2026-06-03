"""
src/download.py
---------------
Downloads OpenNeuro ds004362 (EEG Motor Movement/Imagery) via openneuro-py.

Usage:
    python src/download.py --subjects 10 --output data/
    python src/download.py --all --output data/
    python src/download.py --subject-ids sub-001 sub-002 --output data/
"""

import argparse
import os
import sys
from pathlib import Path

# ── Try importing openneuro-py ────────────────────────────────────────────────
try:
    import openneuro
except ImportError:
    print("[ERROR] openneuro-py not found. Run: pip install openneuro-py")
    sys.exit(1)


DATASET_ID = "ds004362"
DATASET_VERSION = "1.0.0"
PHYSIONET_URL = "https://physionet.org/content/eegmmidb/1.0.0/"

# Subject IDs in the dataset (sub-001 … sub-109)
ALL_SUBJECTS = [f"sub-{i:03d}" for i in range(1, 110)]

# Runs used for left-vs-right imagery (T1=left fist, T2=right fist)
IMAGERY_RUNS = ["run-04", "run-08", "run-12"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download OpenNeuro ds004362 EEG dataset"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--subjects",
        type=int,
        metavar="N",
        help="Download first N subjects (1-109)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Download all 109 subjects (~3 GB)",
    )
    group.add_argument(
        "--subject-ids",
        nargs="+",
        metavar="SUB_ID",
        help="Specific subject IDs, e.g. sub-001 sub-002",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/",
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--imagery-only",
        action="store_true",
        default=True,
        help="Download only imagery runs (4, 8, 12) to save space",
    )
    return parser.parse_args()


def get_subject_list(args) -> list[str]:
    if args.all:
        return ALL_SUBJECTS
    elif args.subjects:
        n = min(args.subjects, 109)
        return ALL_SUBJECTS[:n]
    else:
        return args.subject_ids


def download_via_openneuro(subjects: list[str], output_dir: Path, imagery_only: bool):
    """Download using openneuro-py (requires no account for CC0 datasets)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Downloading {len(subjects)} subject(s) from OpenNeuro {DATASET_ID}")
    print(f"   Output: {output_dir.resolve()}")
    print(f"   Imagery-only mode: {imagery_only}\n")

    # Build include filter for imagery-only mode
    include = None
    if imagery_only:
        include = []
        for sub in subjects:
            for run in IMAGERY_RUNS:
                include.append(f"{sub}/eeg/{sub}_task-*_{run}_eeg.edf")
                include.append(f"{sub}/eeg/{sub}_task-*_{run}_eeg.set")
                include.append(f"{sub}/eeg/{sub}_task-*_{run}_eeg.fdt")
                include.append(f"{sub}/eeg/{sub}_task-*_{run}_events.tsv")
                include.append(f"{sub}/eeg/{sub}_task-*_{run}_eeg.json")
        # Always include dataset-level files
        include += [
            "dataset_description.json",
            "participants.tsv",
            "participants.json",
            "README",
            "CHANGES",
            "*.json",
        ]

    try:
        openneuro.download(
            dataset=DATASET_ID,
            # version=DATASET_VERSION,
            target_dir=str(output_dir),
            include=include,
        )
        print(f"\n✅ Download complete → {output_dir.resolve()}")
    except Exception as exc:
        print(f"\n[WARNING] openneuro-py download failed: {exc}")
        print("Falling back to wget instructions …")
        print_manual_instructions(subjects, output_dir, imagery_only)


def print_manual_instructions(subjects, output_dir, imagery_only):
    """Print manual download commands as fallback."""
    print("\n" + "=" * 60)
    print("Manual Download Instructions")
    print("=" * 60)
    print("\nOption 1 — openneuro CLI (Node.js):")
    print("  npm install -g @openneuro/cli")
    print(f"  openneuro download --dataset {DATASET_ID} --tag {DATASET_VERSION} \\")
    print(f"    --target {output_dir}/")

    print("\nOption 2 — datalad:")
    print(f"  datalad install https://github.com/OpenNeuroDatasets/{DATASET_ID}.git")

    print("\nOption 3 — PhysioNet direct (same data):")
    print(f"  wget -r -N -c -np {PHYSIONET_URL}")

    print("\nOption 4 — Python snippet:")
    print("""
  import urllib.request
  BASE = "https://physionet.org/files/eegmmidb/1.0.0/"
  for sub_num in range(1, 10):
      sub = f"S{sub_num:03d}"
      for run_num in [4, 8, 12]:
          url = f"{BASE}{sub}/{sub}R{run_num:02d}.edf"
          urllib.request.urlretrieve(url, f"data/{sub}R{run_num:02d}.edf")
""")


def verify_download(output_dir: Path, subjects: list[str]):
    """Check that expected files are present."""
    print("\n🔍 Verifying download …")
    missing = []
    found = 0
    for sub in subjects[:5]:  # check first 5
        for run in IMAGERY_RUNS:
            pattern = list(output_dir.rglob(f"{sub}*{run}*eeg.edf")) + \
                      list(output_dir.rglob(f"{sub}*{run}*eeg.set"))
            if pattern:
                found += 1
            else:
                missing.append(f"{sub}/{run}")

    print(f"   Found: {found} EEG files (checked first 5 subjects × 3 runs)")
    if missing:
        print(f"   Missing: {missing[:5]}")
    else:
        print("   ✅ All checked files present")


def main():
    args = parse_args()
    subjects = get_subject_list(args)
    output_dir = Path(args.output)

    print("=" * 60)
    print("  EEG Motor Imagery BCI — Dataset Download")
    print(f"  Dataset : OpenNeuro {DATASET_ID} v{DATASET_VERSION}")
    print(f"  Subjects: {len(subjects)}")
    print(f"  License : CC0 (public domain)")
    print("=" * 60)

    download_via_openneuro(subjects, output_dir, args.imagery_only)
    verify_download(output_dir, subjects)

    print(f"\n📂 Data directory contents:")
    eeg_files = sorted(list(output_dir.glob("**/*.edf")) + list(output_dir.glob("**/*.set")))
    for p in eeg_files[:10]:
        print(f"   {p.relative_to(output_dir)}")
    if len(eeg_files) > 10:
        print("   … (more files)")


if __name__ == "__main__":
    main()
