"""Share one feature contract between training, CLI prediction and API."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Conservative pre-contact baseline: exclude duration and current-call context.
NUMERIC = ["age", "balance", "previous"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "poutcome"]
FEATURES = NUMERIC + CATEGORICAL


def validate_frame(frame):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    selected = frame[FEATURES].copy()
    if selected.isna().any().any():
        raise ValueError("Features cannot contain missing values")
    for name in NUMERIC:
        selected[name] = pd.to_numeric(selected[name], errors="raise")
        if not selected[name].map(lambda x: float('-inf') < x < float('inf')).all():
            raise ValueError(f"{name} must contain finite numbers")
    if not selected.age.between(18, 120).all() or not selected.previous.ge(0).all():
        raise ValueError("Age must be 18–120 and previous contacts must be nonnegative")
    for name in CATEGORICAL:
        if not selected[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{name} must contain nonempty text")
    return selected


def build_pipeline():
    preparation = ColumnTransformer([
        ("numbers", StandardScaler(), NUMERIC),
        ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prepare", preparation),
                     ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
