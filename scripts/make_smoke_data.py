"""Generate deterministic software-check data, never a model-quality benchmark."""
import json
from pathlib import Path
import pandas as pd

customer = json.loads(Path("examples/customer.json").read_text())
rows = [dict(customer, age=20 + i % 60, balance=i * 13, previous=i % 4,
             job="technician" if i % 2 else "admin.", y="yes" if i % 3 == 0 else "no")
        for i in range(300)]
Path("data").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("data/smoke.csv", sep=";", index=False)
print("Generated 300 SYNTHETIC smoke-test rows. Not for evaluation or release.")
