"""Check the local managed-inference HTTP contract, not the AWS service."""
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen
import joblib
from bank_ml.predict import predict

base = "http://127.0.0.1:18080"
customer = json.loads(Path("examples/customer.json").read_text())
with urlopen(base + "/ping", timeout=10) as response:
    assert json.load(response) == {"status": "ready"}
request = Request(base + "/invocations", data=json.dumps(customer).encode(), headers={"Content-Type": "application/json"})
with urlopen(request, timeout=10) as response:
    actual = json.load(response)["subscription_probability"]
expected = predict(joblib.load("release/model.joblib"), [customer])[0]["subscription_probability"]
assert math.isclose(actual, expected, rel_tol=1e-6)
print("PASS: local SageMaker-compatible health, inference and release parity")
