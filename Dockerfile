FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --create-home appuser
COPY artifacts/model.joblib ./artifacts/model.joblib
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "bank_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
