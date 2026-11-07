FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ALLOW_LLM_REQUESTS=false
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --create-home appuser
COPY release ./release
USER appuser
EXPOSE 8000
CMD ["python", "-m", "bank_ml.serve"]
