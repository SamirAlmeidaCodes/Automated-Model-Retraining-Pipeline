"""Drop a noisy copy of existing rows into data/incoming for the next retrain."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=30, help="How many rows to write")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Omit for a new batch each run.")
    args = parser.parse_args()
    if args.rows < 1:
        raise SystemExit("--rows must be at least 1")

    source = ROOT / "data" / "breast_cancer.csv"
    if not source.exists():
        raise SystemExit(f"Sample dataset not found at {source}. Run scripts/generate_sample_dataset.py first.")
    frame = pd.read_csv(source)
    rng = np.random.default_rng(args.seed)
    sample = frame.sample(
        n=args.rows,
        replace=args.rows > len(frame),
        random_state=int(rng.integers(0, 1_000_000)),
    ).copy()
    feature_columns = [column for column in frame.columns if column != "target"]
    scale = frame[feature_columns].std(ddof=0).replace(0, 1.0).to_numpy()
    noise = rng.normal(0.0, 0.05, size=sample[feature_columns].shape) * scale
    sample[feature_columns] = sample[feature_columns].to_numpy() + noise

    incoming = ROOT / "data" / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = incoming / f"batch_{stamp}.csv"
    sample.to_csv(destination, index=False)
    print(destination)


if __name__ == "__main__":
    main()
