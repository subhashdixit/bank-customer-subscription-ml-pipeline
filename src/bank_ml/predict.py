"""Load only trusted model files: joblib is an executable serialization format."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from bank_ml.model import validate_frame


def predict(model, rows):
    frame = validate_frame(pd.DataFrame(rows))
    return [{"subscription_probability": float(p)} for p in model.predict_proba(frame)[:, 1]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("examples/customer.json"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    args = parser.parse_args()
    print(json.dumps(predict(joblib.load(args.model), [json.loads(args.input.read_text())]), indent=2))
