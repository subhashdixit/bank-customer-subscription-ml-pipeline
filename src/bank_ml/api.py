"""Run with: python -m uvicorn bank_ml.api:app --host 0.0.0.0 --port 8000."""
from contextlib import asynccontextmanager
import os
import logging

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from bank_ml.predict import predict
from bank_ml.rag import ProductKnowledge, briefing


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
    try:
        app.state.knowledge = ProductKnowledge()
    except Exception:
        logging.getLogger(__name__).warning("Product knowledge unavailable; numerical predictions remain available")
        app.state.knowledge = None
    yield


app = FastAPI(title="Bank subscription prediction", lifespan=lifespan)


@app.get("/ping", include_in_schema=False)
@app.get("/health")
def health():
    return {"status": "ready"}


@app.post("/invocations", include_in_schema=False)
@app.post("/predict")
def prediction(customer: Customer):
    try:
        return predict(app.state.model, [customer.model_dump()])[0]
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


class BriefingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer: Customer
    product_id: str = Field(min_length=1, max_length=80)
    question: str = Field(default="minimum deposit duration early withdrawal eligibility", min_length=1, max_length=1000)
    use_llm: bool = False


@app.get("/products")
def products():
    if app.state.knowledge is None:
        raise HTTPException(status_code=503, detail="Product knowledge unavailable")
    return {"products": app.state.knowledge.catalog()}


@app.post("/predict/briefing")
def enriched_prediction(request: BriefingRequest):
    if request.use_llm and os.getenv("ALLOW_LLM_REQUESTS", "true").lower() != "true":
        raise HTTPException(status_code=403, detail="LLM calls are disabled on this public demo")
    score = prediction(request.customer)
    knowledge = app.state.knowledge
    if knowledge is None:
        return dict(score, briefing={"mode": "unavailable", "message": "Product knowledge unavailable; original score preserved."})
    if request.product_id not in knowledge.products:
        raise HTTPException(status_code=404, detail="Unknown demonstration product")
    try:
        context = briefing(knowledge, request.product_id, request.question, request.use_llm)
    except Exception:
        logging.getLogger(__name__).warning("Briefing unavailable; numerical prediction preserved")
        context = {"mode": "unavailable", "message": "Briefing unavailable; original score preserved."}
    return dict(score, briefing=context)


# Mount last so API routes retain their own validation and OpenAPI documentation.
from a2wsgi import WSGIMiddleware
from bank_ml.web import create_web_app
app.mount("/", WSGIMiddleware(create_web_app()))
