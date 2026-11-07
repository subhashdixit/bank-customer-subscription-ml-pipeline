"""Verify a running API over real HTTP, including its prediction contract."""
import argparse
import json
import math
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def check(base_url):
    base_url = base_url.rstrip("/")
    with urlopen(base_url + "/health", timeout=10) as response:
        assert json.load(response) == {"status": "ready"}
    customer = json.loads(Path("examples/customer.json").read_text())
    request = Request(base_url + "/predict", data=json.dumps(customer).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        result = json.load(response)
    probability = result["subscription_probability"]
    assert isinstance(probability, (int, float)) and math.isfinite(probability)
    assert 0 <= probability <= 1
    # Compare the actual container result against the model used to build it.
    import joblib
    from bank_ml.predict import predict
    expected = predict(joblib.load("artifacts/model.joblib"), [customer])[0]["subscription_probability"]
    assert math.isclose(probability, expected, rel_tol=1e-6, abs_tol=1e-8), (probability, expected)
    request = Request(base_url + "/predict", data=json.dumps(dict(customer, age=-1)).encode(),
                      headers={"Content-Type": "application/json"})
    try:
        urlopen(request, timeout=10)
    except HTTPError as error:
        assert error.code == 422, error.code
    else:
        raise AssertionError("Invalid customer should return HTTP 422")
    payload = {"customer": customer, "product_id": "demo-12-month", "use_llm": False}
    request = Request(base_url + "/predict/briefing", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        enriched = json.load(response)
    assert enriched["subscription_probability"] == probability
    assert enriched["briefing"]["mode"] == "extractive"
    assert enriched["briefing"]["sources"]
    assert all(s["id"].startswith("demo12-") for s in enriched["briefing"]["sources"])
    print(f"PASS: health, prediction schema/range, model parity, invalid input, grounded product briefing; probability={probability}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    check(parser.parse_args().url)
