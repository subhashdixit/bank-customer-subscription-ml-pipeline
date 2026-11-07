import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from bank_ml import rag
from bank_ml.api import app


def test_product_filtering_and_no_evidence():
    knowledge = rag.ProductKnowledge()
    sources = knowledge.retrieve("demo-12-month", "minimum deposit")
    assert sources[0]["id"] == "demo12-minimum"
    assert all(s["id"].startswith("demo12-") for s in sources)
    assert "10,000" in sources[0]["text"]
    assert knowledge.retrieve("demo-6-month", "minimum deposit")[0]["id"] == "demo6-minimum"
    result = rag.briefing(knowledge, "demo-12-month", "quantum aardvark")
    assert result["mode"] == "no_evidence"
    assert result["sources"] == []


def test_llm_curation_validation_and_fallback(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-not-a-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("LLM_MODEL", "test")
    actual_client = httpx.Client
    outputs = [
        {"source_ids": ["demo12-minimum"], "question_ids": ["amount"]},
        {"source_ids": ["demo6-minimum"], "question_ids": []},
        {"source_ids": ["invented"], "question_ids": []},
        {"source_ids": [], "question_ids": []},
    ]
    def handler(request):
        body = json.loads(request.content)
        assert "balance" not in body["messages"][1]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(outputs.pop(0))}}]})
    monkeypatch.setattr(rag.httpx, "Client", lambda **kwargs: actual_client(transport=httpx.MockTransport(handler), **kwargs))
    k = rag.ProductKnowledge()
    first = rag.briefing(k, "demo-12-month", "minimum deposit", True)
    assert first["mode"] == "llm_curated"
    assert first["product_information"] == [{"text": first["sources"][0]["text"], "source_id": "demo12-minimum"}]
    for _ in range(3):
        fallback = rag.briefing(k, "demo-12-month", "minimum deposit", True)
        assert fallback["mode"] == "extractive"
        assert fallback["llm_status"] == "unavailable_or_invalid_response"


def test_llm_timeout_and_missing_configuration(monkeypatch):
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    k = rag.ProductKnowledge()
    assert rag.briefing(k, "demo-12-month", "minimum", True)["llm_status"] == "not_configured"
    for name, value in (("LLM_API_KEY", "test"), ("LLM_BASE_URL", "https://provider.example/v1"), ("LLM_MODEL", "test")):
        monkeypatch.setenv(name, value)
    actual_client = httpx.Client
    def timeout(request):
        raise httpx.ReadTimeout("simulated", request=request)
    monkeypatch.setattr(rag.httpx, "Client", lambda **kwargs: actual_client(transport=httpx.MockTransport(timeout), **kwargs))
    assert rag.briefing(k, "demo-12-month", "minimum", True)["mode"] == "extractive"


def test_integrated_api_and_failure_isolation(tmp_path, monkeypatch):
    import joblib
    import pandas as pd
    from bank_ml.model import build_pipeline
    customer = json.loads(Path("examples/customer.json").read_text())
    model = build_pipeline().fit(pd.DataFrame([customer, dict(customer, age=65)]), [0, 1])
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    with TestClient(app) as client:
        score = client.post("/predict", json=customer).json()["subscription_probability"]
        payload = {"customer": customer, "product_id": "demo-12-month"}
        response = client.post("/predict/briefing", json=payload)
        assert response.status_code == 200
        result = response.json()
        assert result["subscription_probability"] == score
        assert result["briefing"]["mode"] == "extractive"
        assert len(client.get("/products").json()["products"]) == 2
        assert client.post("/predict/briefing", json=dict(payload, product_id="missing")).status_code == 404
        app.state.knowledge = None
        result = client.post("/predict/briefing", json=payload).json()
        assert result["subscription_probability"] == score
        assert result["briefing"]["mode"] == "unavailable"
        assert client.post("/predict", json=customer).status_code == 200
