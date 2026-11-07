import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from bank_ml.api import app
from bank_ml.serve import verify_release


def test_web_assets_api_and_public_llm_policy(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "release/model.joblib")
    monkeypatch.setenv("ALLOW_LLM_REQUESTS", "false")
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Bank Customer Subscription" in page.text
        assert 'id="prediction-form"' in page.text
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
        assert client.get("/static/app.css").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert "/predict/briefing" in client.get("/openapi.json").json()["paths"]
        customer = json.loads(Path("examples/customer.json").read_text())
        payload = {"customer": customer, "product_id": "demo-12-month"}
        assert client.get("/ping").status_code == 200
        assert client.post("/invocations", json=customer).json() == client.post("/predict", json=customer).json()
        assert client.post("/predict/briefing", json=payload).json()["briefing"]["sources"]
        assert client.post("/predict/briefing", json=dict(payload, use_llm=True)).status_code == 403


def test_release_integrity(tmp_path):
    assert verify_release(Path("release"))["dataset_sha256"] == "d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d"
    for name in ("model.joblib", "metrics.json", "manifest.json"):
        shutil.copyfile(Path("release") / name, tmp_path / name)
    (tmp_path / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_release(tmp_path)
