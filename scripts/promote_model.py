"""Create an explicit, reviewable release from a trusted training run."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, default=Path("artifacts"))
parser.add_argument("--output", type=Path, default=Path("release"))
parser.add_argument("--expected-data-sha256", required=True)
args = parser.parse_args()
metrics = json.loads((args.source / "metrics.json").read_text())
if metrics["data_sha256"] != args.expected_data_sha256:
    raise ValueError("Dataset checksum differs from the explicitly approved dataset")
args.output.mkdir(parents=True, exist_ok=True)
for name in ("model.joblib", "metrics.json"):
    shutil.copyfile(args.source / name, args.output / name)
manifest = {
    "release": "bank-subscription-baseline-v1", "training_data": "UCI bank-full.csv",
    "dataset_sha256": metrics["data_sha256"], "sklearn": metrics["sklearn"],
    "files": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
              for name in ("model.joblib", "metrics.json")},
    "limitations": "Historical educational baseline; not calibrated or approved for business use."
}
(args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
