"""Train on earlier records and evaluate on a later, untouched holdout."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from bank_ml.model import FEATURES, build_pipeline, validate_frame


def train(data_path, output):
    frame = pd.read_csv(data_path, sep=";")
    if "y" not in frame or not frame.y.isin(["yes", "no"]).all():
        raise ValueError("Expected a y column containing only yes/no")
    if len(frame) < 100:
        raise ValueError("At least 100 rows are required")
    features = validate_frame(frame)
    labels = frame.y.eq("yes").astype(int)
    # UCI bank-full is date ordered. Keep the latest 20% as the test set.
    split = int(len(frame) * 0.8)
    if labels.iloc[:split].nunique() != 2 or labels.iloc[split:].nunique() != 2:
        raise ValueError("Both time partitions must contain both target classes")
    model = build_pipeline()
    model.fit(features.iloc[:split], labels.iloc[:split])
    probabilities = model.predict_proba(features.iloc[split:])[:, 1]
    actual = labels.iloc[split:]
    metrics = {
        "train_rows": split, "test_rows": len(actual), "split": "first 80% / final 20%, chronological",
        "test_subscription_rate": float(actual.mean()),
        "average_precision": float(average_precision_score(actual, probabilities)),
        "roc_auc": float(roc_auc_score(actual, probabilities)),
        "precision_at_0_5": float(precision_score(actual, probabilities >= .5, zero_division=0)),
        "recall_at_0_5": float(recall_score(actual, probabilities >= .5, zero_division=0)),
        "training_subscription_rate": float(labels.iloc[:split].mean()),
        "features": FEATURES, "excluded": [name for name in frame.columns if name not in FEATURES + ["y"]],
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "limitations": "Educational baseline; no hyperparameter tuning or probability calibration. Chronological holdout is not a customer-group split."
    }
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/bank-full.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    train(args.data, args.output)
