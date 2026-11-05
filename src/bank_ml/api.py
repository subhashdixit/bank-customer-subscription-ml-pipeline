"""Run with: python -m uvicorn bank_ml.api:app --host 0.0.0.0 --port 8000."""
from contextlib import asynccontextmanager
import os

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from bank_ml.predict import predict


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)
    age: int = Field(ge=18, le=120)
    balance: float
    previous: int = Field(ge=0)
    job: str = Field(min_length=1)
    marital: str = Field(min_length=1)
    education: str = Field(min_length=1)
    default: str = Field(min_length=1)
    housing: str = Field(min_length=1)
    loan: str = Field(min_length=1)
    poutcome: str = Field(min_length=1)


@asynccontextmanager
async def lifespan(app):
    app.state.model = joblib.load(os.environ.get("MODEL_PATH", "artifacts/model.joblib"))
    yield


app = FastAPI(title="Bank subscription prediction", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ready"}


@app.post("/predict")
def prediction(customer: Customer):
    try:
        return predict(app.state.model, [customer.model_dump()])[0]
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
