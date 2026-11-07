"""Production entry point: check the trusted release, then respect hosting PORT."""
import hashlib
import json
import os
from pathlib import Path

import sklearn
import uvicorn


def verify_release(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if sklearn.__version__ != manifest["sklearn"]:
        raise ValueError("Release and runtime scikit-learn versions differ")
    for name in ("model.joblib", "metrics.json"):
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != manifest["files"][name]:
            raise ValueError(f"Release checksum mismatch: {name}")
    return manifest


if __name__ == "__main__":
    release_dir = Path(os.environ.get("RELEASE_DIR", "release"))
    verify_release(release_dir)
    os.environ["MODEL_PATH"] = str(release_dir / "model.joblib")
    uvicorn.run("bank_ml.api:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), workers=1)
