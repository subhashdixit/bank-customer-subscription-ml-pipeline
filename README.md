# Bank customer subscription ML pipeline

Learn how a machine learning application moves from a dataset to predictions, a Docker container, automated checks, and eventually AWS.

**Problem:** a bank wants to prioritise customers for a term-deposit marketing call. We estimate subscription likelihood using information available before contact. This is an educational baseline, not a production banking decision system.

**Cost approach:** development and inference run on your own computer. No AWS resources are provisioned. This repository is public. Standard GitHub-hosted runners are free for public repositories; storage and other runner types have separate billing rules.

## Start here

Read [the beginner handbook](docs/01-beginner-handbook.md) before copying commands. It explains the commands, expected results, each file, and the learning exercises.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m bank_ml.data
python -m bank_ml.train
python -m bank_ml.predict
python -m pytest -q
python -m uvicorn bank_ml.api:app --host 127.0.0.1 --port 8000
```

While the API runs, open http://127.0.0.1:8000/docs to send a prediction. Stop it with Control+C.

## Learning guide

0. [Study guide: architecture, code walkthrough, CI explained and exercises](docs/00-study-guide.md)
1. [Beginner handbook: setup, data, training, predictions, API, Docker, Git and CI](docs/01-beginner-handbook.md)
2. [Data card and model limitations](docs/02-data-and-model.md)
3. [AWS architecture and staged deployment roadmap](docs/03-aws-roadmap.md)
4. [Troubleshooting](docs/04-troubleshooting.md)
5. [Glossary](docs/05-glossary.md)
6. [Actual validation evidence](docs/06-validation.md)

## Current capabilities

- Download UCI data and record source, license and checksum.
- Train a preprocessing + logistic regression pipeline with a chronological holdout.
- Save model and evaluation results; run CLI and API predictions.
- Validate input; test model serialization, API behaviour, leakage exclusion and startup failure.
- Package a trained model in a non-root Docker container.
- GitHub workflow tests, trains, builds and smoke-checks the container.

AWS configuration, S3 integration, model registry, drift monitoring, automated retraining, and continuous deployment are **future stages**. The workflow currently performs CI; it does not deploy a persistent service. See the validation record for what has actually been run.

## Files

| Location | Responsibility |
| --- | --- |
| `src/bank_ml/data.py` | Download source data |
| `src/bank_ml/model.py` | Feature contract and preprocessing |
| `src/bank_ml/train.py` | Training and holdout evaluation |
| `src/bank_ml/predict.py` | CLI and shared prediction function |
| `src/bank_ml/api.py` | HTTP interface |
| `tests/` | Automated checks using generated examples |
| `examples/customer.json` | Fictional prediction request |
| `Dockerfile` | Container build instructions |
| `.github/workflows/ci.yml` | GitHub automation |
| `docs/` | Lessons, limitations and evidence |
| `infra/` | Reserved for later AWS implementation |

Raw data and generated models stay out of Git. Never commit credentials or load model files from untrusted sources.
