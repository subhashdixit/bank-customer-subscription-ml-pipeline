import json
import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from bank_ml.api import app
from bank_ml.model import FEATURES, build_pipeline, validate_frame
from bank_ml.predict import predict
from bank_ml.train import train


@pytest.fixture
def customer():
    return json.loads(open("examples/customer.json").read())


def test_training_roundtrip_and_api(tmp_path, monkeypatch, customer):
    rows = []
    for index in range(200):
        row = dict(customer, age=20 + index % 60, balance=index * 13,
                   job="technician" if index % 2 else "admin.", y="yes" if index % 3 == 0 else "no")
        rows.append(row)
    data = tmp_path / "bank.csv"
    pd.DataFrame(rows).to_csv(data, sep=";", index=False)
    output = tmp_path / "artifacts"
    metrics = train(data, output)
    assert metrics["test_rows"] == 40
    assert "duration" not in metrics["features"]
    model = joblib.load(output / "model.joblib")
    expected = predict(model, [customer])[0]
    assert 0 <= expected["subscription_probability"] <= 1
    monkeypatch.setenv("MODEL_PATH", str(output / "model.joblib"))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ready"}
        assert client.post("/predict", json=customer).json() == expected
        assert client.post("/predict", json=dict(customer, age=-1)).status_code == 422
        assert client.post("/predict", json=dict(customer, duration=99)).status_code == 422
        invalid = dict(customer)
        del invalid["job"]
        assert client.post("/predict", json=invalid).status_code == 422


def test_unknown_category_and_leakage(customer):
    rows = [dict(customer, duration=0), dict(customer, age=60, duration=999)]
    model = build_pipeline().fit(validate_frame(pd.DataFrame(rows)), [0, 1])
    first = predict(model, [dict(customer, duration=0)])
    assert first == predict(model, [dict(customer, duration=999)])
    assert len(predict(model, [dict(customer, job="unseen-job")])) == 1
    assert "duration" not in FEATURES


def test_validation(customer):
    with pytest.raises(ValueError, match="Missing"):
        validate_frame(pd.DataFrame([{"age": 30}]))
    with pytest.raises(ValueError):
        validate_frame(pd.DataFrame([dict(customer, balance=float("inf"))]))


def test_missing_model_fails_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass
