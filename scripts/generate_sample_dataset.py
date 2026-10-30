"""Write the Breast Cancer Wisconsin (Diagnostic) sample files.

The table is the copy bundled with scikit-learn, originally published by
Wolberg, Street, and Mangasarian and distributed by the UCI Machine Learning
Repository. Run this from the repository root when you want to rebuild it:

    python scripts/generate_sample_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dataset = load_breast_cancer()
    frame = pd.DataFrame(dataset.data, columns=list(dataset.feature_names))
    frame["target"] = dataset.target.astype(int)
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "breast_cancer.csv"
    frame.to_csv(csv_path, index=False)

    meta = {
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "target_column": "target",
        "label_map": {"0": "malignant", "1": "benign"},
        "source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "description": (
            "Cell nucleus measurements from digitized fine-needle aspirates. "
            "Class 0 is malignant and class 1 is benign. "
            "Features and labels are the scikit-learn copy of the UCI dataset."
        ),
    }
    meta_path = data_dir / "dataset_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    example = {
        "features": {
            name: float(value)
            for name, value in zip(dataset.feature_names, dataset.data[0], strict=True)
        }
    }
    examples = ROOT / "examples"
    examples.mkdir(parents=True, exist_ok=True)
    example_path = examples / "predict_request.json"
    example_path.write_text(json.dumps(example, indent=2) + "\n", encoding="utf-8")

    label = int(dataset.target[0])
    label_name = meta["label_map"][str(label)]
    print(f"Wrote {csv_path} ({len(frame)} rows)")
    print(f"Wrote {meta_path}")
    print(f"Wrote {example_path} (row 0 label: {label} {label_name})")


if __name__ == "__main__":
    main()
