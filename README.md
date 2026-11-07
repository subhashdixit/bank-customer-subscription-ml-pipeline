# Bank Customer Subscription ML Deployment

A single, step-by-step learning book: business problem, local development, model evaluation, APIs, Docker, CI/CD, AWS deployment design, monitoring, drift, retraining and scaling.

**Start here:** chapters 01–18 cover the currently implemented local/CI system; chapters 19–27 explain the full intended lifecycle and clearly label planned implementations. Chapters 28–32 provide troubleshooting, exercises, next steps, evidence and terminology. No AWS resources are created by reading or running the current local pipeline.

The same prediction service now includes **product-scoped FAISS retrieval and an optional LLM-curated briefing**. Try the enriched endpoint in chapter 33. It does not change the classifier score or retrain the model.

**Web app:** the same application now has a Flask customer form at `/`, alongside the FastAPI explorer at `/docs`. Chapter 34 explains local use and free Render hosting. Chapters 35–38 retain the AWS learning path with concrete ECR, EC2, SageMaker and operations exercises. AWS commands are future, billable exercises, not actions performed by this repository automatically.

## Contents

- [01. Business problem and proposed solution](#01-business-problem-and-proposed-solution)
- [02. Scope, implementation status and cost](#02-scope-implementation-status-and-cost)
- [03. Architecture and the complete lifecycle](#03-architecture-and-the-complete-lifecycle)
- [04. Project files and their responsibilities](#04-project-files-and-their-responsibilities)
- [05. Terminal, Python and environment setup](#05-terminal-python-and-environment-setup)
- [06. Dataset, attribution and prediction timing](#06-dataset-attribution-and-prediction-timing)
- [07. Download and record the data](#07-download-and-record-the-data)
- [08. Customer fields and validation](#08-customer-fields-and-validation)
- [09. Preprocessing and model code](#09-preprocessing-and-model-code)
- [10. Train, evaluate and save the model](#10-train-evaluate-and-save-the-model)
- [11. Metrics, leakage and honest interpretation](#11-metrics-leakage-and-honest-interpretation)
- [12. Predict from the command line](#12-predict-from-the-command-line)
- [13. Serve predictions over HTTP](#13-serve-predictions-over-http)
- [14. Tests and failure behaviour](#14-tests-and-failure-behaviour)
- [15. Docker build and container execution](#15-docker-build-and-container-execution)
- [16. Git, GitHub and change history](#16-git-github-and-change-history)
- [17. CI pipeline explained step by step](#17-ci-pipeline-explained-step-by-step)
- [18. Verify the complete implemented pipeline](#18-verify-the-complete-implemented-pipeline)
- [19. Batch prediction and new data design](#19-batch-prediction-and-new-data-design)
- [20. Model versions, approval and release design](#20-model-versions-approval-and-release-design)
- [21. AWS foundations and service responsibilities](#21-aws-foundations-and-service-responsibilities)
- [22. EC2 deployment path](#22-ec2-deployment-path)
- [23. SageMaker training and endpoint path](#23-sagemaker-training-and-endpoint-path)
- [24. Continuous delivery, deployment and rollback](#24-continuous-delivery-deployment-and-rollback)
- [25. Monitoring, data drift and delayed outcomes](#25-monitoring-data-drift-and-delayed-outcomes)
- [26. Retraining and improvement decisions](#26-retraining-and-improvement-decisions)
- [27. Scaling, reliability and cost decisions](#27-scaling-reliability-and-cost-decisions)
- [28. Troubleshooting and recovery](#28-troubleshooting-and-recovery)
- [29. Exercises and interview checkpoints](#29-exercises-and-interview-checkpoints)
- [30. Implementation sequence from here](#30-implementation-sequence-from-here)
- [31. Validation evidence and measured limitations](#31-validation-evidence-and-measured-limitations)
- [32. Glossary](#32-glossary)
- [33. Integrated product retrieval and optional LLM briefing](#33-integrated-product-retrieval-and-optional-llm-briefing)
- [34. Flask web app and free Render deployment](#34-flask-web-app-and-free-render-deployment)
- [35. AWS preparation, identity, storage and container registry](#35-aws-preparation-identity-storage-and-container-registry)
- [36. AWS EC2 deployment lab](#36-aws-ec2-deployment-lab)
- [37. SageMaker training and inference lab](#37-sagemaker-training-and-inference-lab)
- [38. Cloud CI/CD, monitoring, recovery and cleanup](#38-cloud-cicd-monitoring-recovery-and-cleanup)

## 01. Business problem and proposed solution

A bank offers a term deposit: a customer places money with the bank for an agreed period. The bank has historical marketing results and wants to prioritise future contacts. Our model estimates subscription likelihood before the current call.

The input is one customer's known information. The output is a number between 0 and 1. The application does not contact customers, approve loans, move money, or guarantee a subscription. The example JSON is fictional. Our model is a learning baseline, not an approved banking system.

“End to end” has a specific scope here: downloading data, training, evaluating, saving, loading, serving through HTTP, packaging in Docker, and automatically checking those steps on GitHub. Persistent cloud deployment is not implemented yet. See the final section for the remaining learning path.

In a real team, the business owner defines calling capacity and costs; a data engineer supplies reliable records; a data scientist evaluates models; and an ML/platform engineer deploys and operates the system. One person may cover several roles in a small team. This project lets you practise those responsibilities at learning scale.

A useful objective is “find more subscribers within the same calling capacity.” If agents can call only 500 customers tomorrow, ranking the daily list and evaluating precision/recall at that capacity may be more useful than applying a universal 0.5 threshold. Predicting subscription is not the same as predicting the extra benefit caused by calling: that would require a different causal/uplift design and suitable evidence.

## 02. Scope, implementation status and cost

This README is the single learning document. Read it from top to bottom; the contents below are shortcuts within this same file. Commands are tested for this macOS/Linux project setup. Do not paste every code block at once: servers occupy a terminal until stopped.

| Capability | Status |
| --- | --- |
| Public data download, preprocessing, training, evaluation | Implemented and tested |
| Saved model, command-line prediction, FastAPI service | Implemented and tested |
| Docker packaging and GitHub CI with real HTTP checks | Implemented and tested |
| FAISS product retrieval, cited excerpts, optional constrained LLM curation | Implemented; live provider call requires configuration |
| Flask form, checksummed baseline release, Render configuration | Implemented; live hosting evidence recorded below |
| Batch scoring CLI and managed model registry | Planned |
| EC2 CloudFormation template and SageMaker container wrappers | Implemented locally; real AWS verification pending |
| Monitoring, delayed-label evaluation, drift and retraining | Planned |

“Implemented” means executable project code exists. “Planned” means the concepts are explained here but working implementation and verification remain to be done. Chapters 35–38 provide AWS lab instructions and optional infrastructure files. They have not been applied to a real AWS account; permissions, quotas, networking and regional availability must be verified when you execute them.

No AWS resources have been created. Current training and serving run on your laptop, and CI runs on temporary GitHub runners. This repository is public; standard hosted runners for public repositories are free, while storage and other runner types have separate billing rules. Actual EC2/SageMaker usage cannot be guaranteed free without credits or a suitable sandbox. A budget alert is not a spending cap.

## 03. Architecture and the complete lifecycle

```mermaid
flowchart TD
    U[Public UCI archive] --> D[data.py: download CSV and provenance]
    D --> T[train.py: chronological split]
    T --> P[Fit preprocessing and classifier on training rows]
    P --> E[Evaluate on later holdout rows]
    E --> M[model.joblib and metrics.json]
    M --> C[predict.py: command-line prediction]
    M --> A[api.py: load model at startup]
    J[Customer JSON] --> C
    J --> A
    A --> R[HTTP probability response]
    M --> I[Docker image containing application and model]
    I --> S[Container serving the same API]
```

**Training path:** many labelled rows become a fitted model and a report. Training is typically a separate job that finishes and exits.

**Serving path:** a running program receives a customer, checks it, transforms it with saved preprocessing, and predicts. It keeps running to handle more requests. It does not train on incoming requests.

**Automation path:** GitHub starts a temporary computer that repeats tests, synthetic smoke training and container checks. An optional manual job performs real-data training. That computer is removed after the job. It is not a permanent prediction server.

The intended full lifecycle is: business goal → versioned raw data → validation → training/validation/test → evaluated candidate → approval → versioned release → serving → monitoring → delayed outcomes → investigation → candidate retraining. Monitoring feeds back into development; it is not the last task before forgetting the model.

## 04. Project files and their responsibilities

| File or directory | Why it exists | Committed to Git? |
| --- | --- | --- |
| `pyproject.toml` | Package metadata and dependency requirements | Yes |
| `src/bank_ml/` | Importable application code | Yes |
| `tests/` | Repeatable checks with synthetic data | Yes |
| `scripts/check_api.py` | Smoke-check numerical and enriched predictions over HTTP | Yes |
| `src/bank_ml/rag.py` | FAISS retrieval, evidence rendering, optional LLM curation | Yes |
| `src/bank_ml/knowledge/deposits.json` | Fictional versioned product passages | Yes |
| `scripts/evaluate_retrieval.py` | Small retrieval regression set | Yes |
| `.env.example` | Placeholder provider configuration, never real secrets | Yes |
| `examples/customer.json` | Fictional valid input | Yes |
| `data/` | Downloaded raw CSV and provenance | Only its placeholder |
| `artifacts/` | Experiment outputs, regenerated by training | No |
| `release/` | Explicitly promoted, checksummed baseline model and evidence | Yes, small educational release snapshot |
| `render.yaml` | Free Render service definition | Yes |
| `infra/aws/` | Optional, billable EC2/SageMaker lab files | Yes |
| `src/bank_ml/web.py`, `templates/`, `static/` | Flask page and browser assets | Yes |
| `docs/baseline-metrics.json` | Explicit historical learning evidence | Yes |
| `.venv/` | Machine-specific Python environment | No |
| `.git/` | Local repository history and configuration | Managed by Git, never stage as application data |
| `.gitignore` | Prevent generated and sensitive files being staged by default | Yes |
| `.dockerignore` | Limit files sent to Docker build | Yes |
| `.github/workflows/ci.yml` | Instructions GitHub executes | Yes |
| `infra/` and `notebooks/` | Reserved for future infrastructure and exploration | Empty folders may not appear on GitHub |

The `src/` layout separates importable code from the README and generated files. Installing the package lets `python -m bank_ml.train` find it without manually changing Python's import path.

## 05. Terminal, Python and environment setup

The terminal is an application where you type commands. Its current directory determines where files are read and written. Open a terminal in this project's root folder, the folder containing `pyproject.toml`.

On macOS, `pwd` prints the current folder and `ls` lists its contents. A command printed in a code block is something you can run; a filename in a paragraph is something you can open and read.


For a fresh clone, first run:

```bash
git clone https://github.com/subhashdixit/bank-customer-subscription-ml-pipeline.git
cd bank-customer-subscription-ml-pipeline
```

If you are already in the existing checkout, use it rather than cloning a second copy. On Windows, shell activation and path syntax differ; the current instructions and evidence cover macOS/Linux.

You need Python 3.10 or newer. Docker is needed only for the container lesson. Git records changes; GitHub hosts the repository online. You do not need an AWS account for these lessons.

```bash
python3 --version
pwd
ls
```

Expected: a supported Python version and project files. If a command is missing, use the troubleshooting guide rather than installing random packages.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

`python3 -m venv .venv` creates a private set of Python tools in `.venv`. It does not create a virtual machine. `source` activates those tools in this terminal; activate again when you open another terminal. `python -m` runs a module using the selected Python interpreter.

`pip` installs libraries. `-e` means editable: when you change our Python source, the environment sees the changes. The quoted `.[dev]` means this project plus its development tools. Quotes prevent the shell treating brackets as filename patterns. `pyproject.toml` declares those requirements. Version ranges allow compatible updates; this is not a fully locked dependency environment.

Expected: installation finishes without an error. The `.venv` directory is ignored by Git because it is generated and specific to your machine. Do not use `sudo pip`.


## 06. Dataset, attribution and prediction timing

### Source and attribution

Dataset: Bank Marketing, UCI Machine Learning Repository, dataset 222.

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing

Download: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip

Creators credited by UCI: S. Moro, P. Rita, P. Cortez. Dataset DOI: https://doi.org/10.24432/C5K306

UCI lists the license as Creative Commons Attribution 4.0 International: https://creativecommons.org/licenses/by/4.0/

We use `bank-full.csv`, the original date-ordered 45,211-row version, not the 41,188-row `bank-additional-full.csv`. This project selects features and derives a binary label; it does not redistribute the raw dataset. Keep this attribution when reusing it.

### Prediction timing and features

The score is calculated before the current marketing contact. Features: age, balance, previous contact count, job, marital status, education, credit default, housing loan, personal loan, and previous campaign outcome. `unknown` is an explicit source category, not a missing Python value. New categories are encoded as all-zero indicators for that feature; prediction continues but may be less reliable.

We exclude duration because it is known only after the call. We also omit contact channel, current campaign count, day and month for a conservative first baseline. The source does not establish that every financial field was frozen at the exact scoring time. Historical leakage and repeated-customer effects therefore remain limitations. This is not a causal model.

### Evaluation

First 80% train, final 20% test, preserving source order. Do not shuffle these rows. This better resembles prediction on later campaigns than a random split, but record-level timestamps and customer IDs are absent from our selected source. The same person may appear across periods. No tuning is performed, and threshold 0.5 is only illustrative.

Average precision and ROC AUC describe ranking. Precision/recall at a fixed threshold describe a particular decision rule. The test positive rate is the random-ranking reference for average precision. Accuracy alone is not useful for this imbalanced target. Generated test fixtures validate code, not measured business performance.

### Intended use and limits

Educational demonstration only. No fairness audit, privacy review, probability calibration, modern population validation, or business cost analysis has been performed. Historical Portuguese bank marketing outcomes do not establish generalisation to other banks or countries. Age and marital information can raise important policy and fairness questions that must be assessed before operational use.

`joblib` model files must come from trusted builds because loading them can execute code. Model and preprocessing are serialized together. A different library version may be incompatible, so metrics record versions. Deployment should eventually use pinned dependencies, immutable model versions and explicit approval gates.

## 07. Download and record the data

```bash
python -m bank_ml.data
```

The program downloads a public ZIP from UCI, reads the original `bank-full.csv` from its nested ZIP, and saves it in `data/`. It also writes a `.source.json` file with a source URL, license and SHA-256 checksum. The checksum is a fingerprint: different bytes produce a different fingerprint. It records what we downloaded; it is not an independently verified authenticity signature.

Expected: `data/bank-full.csv` and `data/bank-full.source.json`. The CSV uses semicolons, so our reader explicitly uses `sep=";"`. This is real historical learning data, not a random fixture. Tests use generated examples so ordinary tests do not need internet access.

Read chapter 06 before changing the feature list. Exercise: open the CSV and locate `y`, the answer column, and `duration`, a column we deliberately exclude.

### `data.py`: bytes become a dataset

`URL` identifies the UCI download. `urlopen` performs an HTTPS request with a timeout. `ZipFile` reads the outer archive and the nested `bank.zip`. The code selects `bank-full.csv` and writes those bytes, rather than extracting arbitrary file paths from an archive. If UCI changes its archive structure, this code can fail and needs review.

`hashlib.sha256` fingerprints the CSV. The `.source.json` records that fingerprint, source and license. It does not independently authenticate the dataset or freeze future downloads. A rerun replaces these local output files. The loader reads the archive into memory, which is reasonable for this small dataset, not a general large-data download design.

## 08. Customer fields and validation

| Field | Type | Meaning in this source |
| --- | --- | --- |
| `age` | Integer, 18–120 in the API | Age in years |
| `balance` | Finite number | Average yearly balance in euros; negative values are permitted |
| `previous` | Nonnegative integer in the API | Contacts before the current campaign |
| `job` | Nonempty text | Job category, such as technician |
| `marital` | Nonempty text | Marital status category |
| `education` | Nonempty text | Education category |
| `default` | Nonempty text | Whether credit is in default |
| `housing` | Nonempty text | Whether a housing loan exists |
| `loan` | Nonempty text | Whether a personal loan exists |
| `poutcome` | Nonempty text | Previous campaign outcome, including unknown |

These definitions describe the historical dataset; they are not a recommended production feature policy. See chapter 06 for fairness and timing limitations. `y` is the training answer, yes/no. It must not be submitted as a prediction feature.

The API forbids extra fields and validates types and ranges with Pydantic. The internal DataFrame validator selects known features and checks missing, finite and basic range values. It intentionally drops other CSV columns. These interfaces are not identical: the internal validator accepts some numeric conversions and does not require integer counts; the public API has integer fields. Categorical fields are not restricted to a fixed vocabulary; unseen categories are allowed and may reduce reliability.

## 09. Preprocessing and model code

### `model.py`: establish a shared contract

`NUMERIC` and `CATEGORICAL` define the feature lists. `validate_frame` selects those fields, rejects missing values, converts numeric values and checks finiteness and ranges. A pandas DataFrame is a table with named columns, similar to a spreadsheet held in memory.

`StandardScaler` learns each numeric column's mean and spread on training data. `OneHotEncoder` represents text categories with indicator columns. For example, a job could become separate technician and administrator indicators. `ColumnTransformer` sends each column group through the appropriate transformation.

`Pipeline` connects preparation and logistic regression into one object. `fit` learns preparation settings and model weights; `predict_proba` applies them to new rows. Logistic regression combines transformed values into a score and maps it to a value between zero and one. It is a classification algorithm despite its name.

## 10. Train, evaluate and save the model

```bash
python -m bank_ml.train
```

The program reads the data, checks required columns and values, and selects a conservative set of pre-contact features. It keeps the earlier 80% of records for training and later 20% for testing because the source file is date ordered.

Numeric features are scaled so a large balance does not have a large numerical influence merely because it uses larger units. Text categories become numeric indicators through one-hot encoding. Both operations are fitted on training rows only. A logistic regression classifier then learns a score from those transformed values.

These operations are saved together as a scikit-learn Pipeline. A new customer must receive exactly the same transformations as training customers. Saving just the classifier would lose that preparation.

Outputs:

- `artifacts/model.joblib`: fitted preprocessing and model.
- `artifacts/metrics.json`: evaluation, feature list, source checksum and software versions.

Read average precision alongside the test subscription rate. A random ranking has expected average precision around the positive rate, though finite samples vary. ROC AUC measures ranking across thresholds. Precision answers “of the customers flagged, how many subscribed?” Recall answers “of the subscribers, how many were flagged?” We report precision and recall at 0.5 as a demonstration, not a business-approved calling threshold.

We fix the model settings before evaluating the holdout. There is no tuning or validation set in this first baseline. If we later tune settings, we must add a validation period and preserve a fresh final test period. Repeatedly adjusting the model based on test results turns the test set into development data.

Exercise: find train/test row counts in the JSON. Explain why the test rows must not be used to fit the scaler.

### Training code walkthrough


`argparse` reads optional `--data` and `--output` command arguments. `Path` handles file locations. Pandas loads the semicolon-separated CSV. The code checks yes/no labels, a minimum row count and that both chronological partitions contain both outcomes.

The first 80% is training data; the final 20% is the holdout. The scaler, encoder and classifier learn only from training rows. `predict_proba(... )[:, 1]` selects the probability for class 1, which is subscription because yes was converted to 1 and no to 0.

Metrics describe the holdout. `joblib.dump` serializes the fitted pipeline so another process can load it. The JSON report records selected/excluded columns, source checksum and software versions. Training currently overwrites the selected output directory's model and report. Use a different `--output` directory to keep an experiment separate, but remember that the API must then be pointed at its model.

## 11. Metrics, leakage and honest interpretation

Suppose 100 customers include 10 actual subscribers. A model flags 8 people and 4 of them are subscribers. Precision is 4/8 = 50%; recall is 4/10 = 40%. Those are different questions. Calling more people may increase recall while reducing precision; the business must choose the trade-off using validation data and costs.

| Report field | Meaning | Limitation |
| --- | --- | --- |
| `test_subscription_rate` | Fraction of yes labels in holdout | Describes this period only |
| `average_precision` | Summary of precision across recall levels | Depends on class prevalence |
| `roc_auc` | Ranking quality across thresholds | Does not establish useful calling economics |
| `precision_at_0_5` | Fraction of flagged customers who subscribed | Depends on the chosen threshold |
| `recall_at_0_5` | Fraction of subscribers flagged | Can be low even when precision is high |

Read the measured [validation record](#31-validation-evidence-and-measured-limitations). It reports weak recall at 0.5 and a large subscription-rate difference between time periods. There is no automatic model-quality gate, tuning, probability calibration, or fairness evaluation. CI passing proves the implemented technical checks passed, not that the model is suitable to use in a bank.

Fitting preprocessing on all rows leaks information about later records. Including call duration leaks knowledge unavailable before the call. Tuning repeatedly against holdout scores leaks evaluation feedback into development decisions.

The current fixed baseline uses no validation set because it does no tuning. Before comparing many models, create separate chronological training, validation and test periods. Model selection belongs on validation data; final evaluation belongs on untouched test data. Date order is a useful approximation here, but missing customer identifiers limit our ability to exclude repeated customers across periods.

## 12. Predict from the command line

```bash
python -m bank_ml.predict --input examples/customer.json
```

JSON is a text format made of named fields and values. Our example customer is fictional. The program loads the saved model, checks the inputs, and prints a probability between zero and one. It does not retrain.

The probability is uncalibrated and based on this historical training distribution. Do not interpret a score as guaranteed real-world odds.

Exercise: copy the example into another local JSON file and change a field. Run the prediction with `--input` pointing at that file. A changed result is an association learned by the model, not proof that changing that characteristic causes subscription.

### Prediction code walkthrough


The CLI reads one customer object from JSON and wraps it in a list. The shared function also accepts multiple row objects internally, but no batch-file CLI is implemented. It validates a DataFrame and returns one probability object per row.

`joblib.load` must receive a trusted file: serialized Python objects can execute code when loaded. The saved model includes preprocessing; do not manually rescale API inputs yourself.

## 13. Serve predictions over HTTP

```bash
python -m uvicorn bank_ml.api:app --host 127.0.0.1 --port 8000
```

Uvicorn is the server process. `bank_ml.api:app` identifies the application object in our code. `127.0.0.1` means your own computer; `8000` is the listening port, like a numbered door for network requests. Keep this terminal open while you use the API.

Open http://127.0.0.1:8000/docs in a browser. FastAPI generates this interactive interface from our input schema. Expand `POST /predict`, choose “Try it out”, paste the example JSON, and execute.

In a second terminal opened in the project root, you can also run:

```bash
curl http://127.0.0.1:8000/health
curl -H 'Content-Type: application/json' --data @examples/customer.json http://127.0.0.1:8000/predict
```

`curl` sends HTTP requests. `--data @file` reads the request body from a file and uses POST. `Content-Type` tells the server how to read the body. GET `/health` asks whether the process is ready. POST `/predict` submits customer information. HTTP 200 means success; 422 means invalid input.

The model loads once at startup. Missing model files stop startup instead of returning fabricated predictions. Stop the server with Control+C. This development API has no authentication or rate limits and should not be exposed publicly.

### API code walkthrough


`Customer` defines the request body schema. FastAPI uses it to validate input and generate `/docs`. `lifespan` loads the model once at startup using `MODEL_PATH`, or its default path. If loading fails, startup fails. The application never silently substitutes an untrained model.

GET `/health` returns readiness for this simple application. It does not assess model quality or recheck disk contents on every request. POST `/predict` uses the already loaded model and converts ordinary validation failures to HTTP 422. Updating a model file on disk does not reload it into an already running process; restart that process or deploy a new container.

## 14. Tests and failure behaviour

```bash
python -m pytest -q
```

Tests are programs that check expected behaviour. We test the full training/save/load/predict path, valid and invalid API requests, unfamiliar job categories, exclusion of call duration, and failure when the model is missing. These tests check software behaviour, not whether the model is commercially useful or fair.

Exercise: send an age of -1 to the API. Explain why it is better to reject it than silently produce a score.

The tests generate synthetic labelled rows, train into a temporary directory and load the resulting model. The API tests use an in-process client; the later smoke test sends real HTTP requests to a running container. Both are useful because an in-process test does not prove the Docker entry point or network path works.

These are software checks. Model-quality gates, load tests and fairness evaluation are different checks. Chapter 33 adds retrieval, LLM-response validation and graceful-failure tests; the historical four-test record in chapter 31 describes the original baseline.

## 15. Docker build and container execution

The image uses the committed real-data release snapshot, so it can build from a fresh clone without downloading UCI. Start Docker Desktop and wait until its engine is ready.

```bash
docker build -t bank-ml:local .
docker run --rm --name bank-ml -p 127.0.0.1:8000:8000 bank-ml:local
```

The build reads `Dockerfile`, installs the application, and includes the trained model. The image is a packaged template; the container is a running instance. `.dockerignore` limits which local files enter the build context. The application runs as a non-root user.

The port mapping forwards laptop port 8000 to container port 8000. The server listens on `0.0.0.0` inside the container so Docker can reach it; the published laptop port is restricted to localhost. Stop any Python API already using that port first.

Run the same curl requests. Stop the container with Control+C. `--rm` removes this stopped container; the reusable image remains. Retraining a local model does not alter a release. Review and promote it with `scripts/promote_model.py`, then rebuild to include the new model; chapter 34 explains this boundary.

A container provides a consistent application environment. It does not automatically provide IAM permissions, TLS, monitoring or production security.

| Instruction | Effect |
| --- | --- |
| `FROM python:3.11-slim` | Start with a Linux image containing Python |
| `WORKDIR /app` | Set the directory for later instructions and startup |
| `COPY pyproject.toml` and `COPY src` | Include package definition and source |
| `RUN pip install ...` | Install runtime dependencies during build |
| `useradd` and `USER appuser` | Run the final service without root privileges |
| `COPY release ./release` | Include the explicitly promoted model, metrics and checksum manifest |
| `EXPOSE 8000` | Document the intended container port; does not publish it |
| `CMD python -m bank_ml.serve` | Verify the release, read hosting PORT and start Uvicorn |

The committed release snapshot was trained before this build; deployment does not retrain it. ECR is not involved locally. `docker run -p 127.0.0.1:8000:8000` publishes the port to your laptop only. Inside the container, Uvicorn listens on all container interfaces so Docker forwarding can reach it. The image bundles a model; changing the host file afterwards does not change that image.

The base-image tag and dependency ranges can change over time. This project has not yet locked every dependency or pinned the base-image digest. A production release should capture those identities and train using the same environment used for serving. Successful cross-version loading in one test is not a general compatibility promise.

## 16. Git, GitHub and change history

Git is local history. GitHub stores a remote copy and runs automation. A commit is a named snapshot of tracked files. Generated data, environments, models and credentials should not be in those snapshots.

```bash
git status
git diff
git add src docs tests README.md
git commit -m "Explain the change you made"
git push
```

Read `git diff` before staging. `git add` selects changes; `git commit` saves them locally; `git push` sends commits to the configured remote. The first publication is handled as part of project setup. Do not rerun repository creation commands for every change.

## 17. CI pipeline explained step by step

The current job uses Ubuntu 24.04 and Python 3.11. Routine CI uses generated synthetic data to avoid depending on UCI availability; it verifies software, not real predictive performance. GitHub's action runtime is separate from application Python: checkout/setup actions use Node.js internally, while our model runs in Python.

| Workflow component | What happens | Typical failure |
| --- | --- | --- |
| `on` | Start on push, pull request or manual dispatch | Actions disabled or YAML invalid |
| `permissions` | Give automation read access to repository contents | Unneeded write operations are rejected |
| `runs-on` | Allocate a temporary Linux runner | Runner service unavailable |
| `timeout-minutes` | Stop a job that exceeds 15 minutes | Slow dependency or data service |
| Checkout action | Retrieve this commit's files | Repository access or action service issue |
| Setup Python action | Select Python 3.11 | Runtime download problem |
| Install | Build our package and install dependencies | Dependency resolution/network error |
| Pytest | Check code using generated data | Behaviour differs from assertions |
| Generate smoke data | Create deterministic synthetic customer rows | Broken fixture script |
| Train | Fit a synthetic smoke-test model, not a release | Schema/label error |
| Predict | Load artifact and score example | Model or input mismatch |
| Build | Package application and model | Missing artifact or failed installation |
| Smoke check | Start container and probe actual HTTP requests | Startup, routing, contract or model mismatch |

The readiness loop tries `/health` up to 30 times, waiting two seconds between attempts. Once ready, `scripts/check_api.py` checks health JSON, a finite probability in range, numerical agreement with the host model, and rejection of an invalid age. A trap prints container logs and removes the temporary container when the shell exits, including when a check fails. Model comparison uses a small numerical tolerance to allow tiny floating-point differences.

The smoke script also checks `/predict/briefing`: the numerical score must match `/predict`, and citations must belong to the requested product. The workflow runs six small retrieval regression checks before training.

The smoke script accepts `--url` for another local port and expects the matching host model. For the standard Docker image run it with `MODEL_PATH=release/model.joblib`; for a direct development server using a newly trained artifact, its default is `artifacts/model.joblib`. A mismatch can indicate that the image contains an older model. It is stronger than merely checking that HTTP returned success.

This is continuous integration. There is no registry push, persistent deployment, release approval, rollback automation or AWS authentication yet. The workflow does not retain its trained model as a downloadable release artifact. CI currently retrains on every triggering change, including documentation pushes unless explicitly skipped. This is simple for learning but inefficient for a larger project.

Updated action runtime references: [checkout](https://github.com/actions/checkout), [setup-python](https://github.com/actions/setup-python). Billing reference: [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions).

## 18. Verify the complete implemented pipeline

Follow chapters 05, 07 and 10 to set up, download and train first. From the activated project environment:

```bash
python -m pytest -q
python -m bank_ml.train
python -m bank_ml.predict
docker build -t bank-ml:local .
docker run --rm --name bank-ml -p 127.0.0.1:8000:8000 bank-ml:local
```

In another terminal at the project root:

```bash
source .venv/bin/activate
MODEL_PATH=release/model.joblib python scripts/check_api.py
```

Expected: a PASS message describing the checked behaviour. Stop the foreground container with Control+C. The image remains available for reuse. If port 8000 is occupied, map laptop port 18000 instead and run the smoke script with `--url http://127.0.0.1:18000`.

After pushing code, open GitHub → repository → Actions → Test and build → latest run. Check its commit matches your change, expand failing steps, and read the first meaningful error. A green older run does not validate newer untested code. Documentation-only commits can use `[skip ci]`; do not use that to bypass verification of code changes.

## 19. Batch prediction and new data design

**Planned implementation.** For this bank problem, a daily ranked calling list is a natural product. Online serving is useful when another application needs an immediate score; batch serving scores a whole file on a schedule.

A batch workflow should receive a dated customer snapshot, validate it, load an approved model once, score in manageable chunks, and write customer identifiers, scores, model version and scoring timestamp. Sort eligible customers by score and apply calling-capacity and business rules separately. The current Python prediction function can score several row objects, but a batch-file CLI, stable customer identifiers and scheduled delivery are not implemented.

New input data and new labelled training data are different. A customer can be scored today without knowing the outcome. Preserve source/event time so tomorrow's information cannot accidentally enter today's features. Keep raw data immutable, write cleaned data separately, quarantine invalid rows with reasons, and record schema versions and checksums. Reprocessing the same batch should not create duplicate outputs; this is called idempotency.

When actual outcomes arrive, join them to prediction records using a stable identifier and an agreed observation window. Do not automatically label an unfinished campaign as “no.” Labels may be missing or delayed, and customers whom the bank never calls may have no comparable observed outcome. Selection bias limits conclusions from those records.

## 20. Model versions, approval and release design

A small local release manifest is implemented in `release/`; approval automation and a managed registry remain planned. A model registry is a catalogue of candidates and their evidence, not just a folder of arbitrary filenames. A local manifest can teach this concept before connecting to a managed registry.

A release should identify the source commit, dataset checksum/version, feature schema, training configuration, dependency environment, model checksum, evaluation report and container image digest. A label such as “latest” is not enough to reproduce a release.

Use candidate → evaluated → approved → deployed states. Compare a new candidate against the current model on the same appropriate validation period. Reject broken schemas, unacceptable performance and missing evidence. Do not use a final test set repeatedly as the promotion score. Define business thresholds before examining the final result.

Development is where changes happen; staging rehearses release checks; production serves real users. They should have separate data and permissions. A release changes only after approval. Keep the previous image/model identity so rollback can restore a known combination. Merely overwriting `model.joblib` loses that history; the current baseline still overwrites its selected output directory.

## 21. AWS foundations and service responsibilities

**Conceptual design; cloud implementation is planned.** AWS is a collection of services with separate responsibilities. Deploying “to AWS” means choosing, securing and connecting them.

| Concept/service | Simple meaning | Role here |
| --- | --- | --- |
| Account and region | Administrative boundary and geographic service location | Choose where resources and billing live |
| IAM role and policy | An assumable identity and allowed actions | Let a job read specific data and write specific outputs |
| S3 bucket and object key | Object storage container and item name | Raw data, prepared data, models, reports |
| ECR | Container image registry | Store versioned images used by compute |
| EC2 and EBS | Virtual machine and attached block storage | Self-managed serving and its disk |
| VPC and subnet | Private network and a segment within it | Place compute and control routing |
| Security group | Stateful network access rules attached to resources | Limit who can reach the service |
| SageMaker AI | Managed ML training and hosting capabilities | Jobs, model registration and inference hosting |
| CloudWatch | Logs, metrics and alarms | Detect operational problems |
| EventBridge or a scheduler | Trigger work at a time or on an event | Batch runs and monitoring jobs |
| DNS and TLS | Service naming and encrypted connections | Stable HTTPS address for an application |
| Infrastructure as code | Versioned resource definitions | Reproducible environment setup and teardown |

A security group controls network traffic; an IAM policy controls authorised API operations. Neither replaces the other. A public S3 bucket is not required for an application to read data: give its role narrowly scoped access. An EC2 security group is not API authentication. An internet-facing service also needs an appropriate authentication and TLS design.

The proposed GitHub deployment uses OIDC identity federation to obtain temporary AWS credentials. The role's trust policy should restrict which repository, branch or environment can assume it; its permission policy should restrict resources and actions. Do not place long-lived AWS keys in source files. Secrets, encryption and access logging need explicit design before handling real customer data.

These explanations introduce the AWS concepts needed for this project; they are not coverage of every AWS service. [AWS IAM introduction](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html), [S3 introduction](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html), [EC2 concepts](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html).

## 22. EC2 deployment path

**Planned implementation.** EC2 gives us a computer; we manage the application on it.

1. Build and test a specific image, then publish it to ECR.
2. Provision a suitably sized instance, disk, network and narrowly scoped instance role through infrastructure as code.
3. Configure the machine to retrieve the image without embedding credentials. Pull the selected immutable image version.
4. Start the container under a service supervisor with restart behaviour and bounded resource/log usage.
5. Provide a controlled ingress path with TLS and authentication as needed. Keep management access restricted.
6. Check health and predictions before directing users to the new release. Record the image/model identity.
7. Send logs and metrics to monitoring, then rehearse restart and rollback.
8. Remove resources when the exercise ends and verify there are no remaining chargeable resources.

For higher availability, more than one instance behind a load balancer can distribute requests, while an Auto Scaling group adjusts capacity. That adds cost and operational complexity; it is not needed for the first local milestone. A single EC2 VM remains a single point of failure. Updating dependencies, operating-system patches, service restart and deployment safety remain our responsibilities.

## 23. SageMaker training and endpoint path

**Planned implementation.** A real SageMaker endpoint is a managed AWS resource. We can implement compatible container behaviour locally, but that cannot verify the managed service, IAM, networking or autoscaling.

A managed training workflow selects training input in S3, a container image, compute settings and an execution role. SageMaker starts the job, makes data available according to the configured input mode, runs training and collects the output. For a custom training container, standard locations include input channels under `/opt/ml/input/data/` and saved model files under `/opt/ml/model`. These differ from our current `data/` and `artifacts/` defaults and need adapters before claiming compatibility.

For standard custom real-time inference, the container serves on port 8080 and supports `/ping` health checks and `/invocations` inference requests. SageMaker places extracted model artifacts under `/opt/ml/model`. The standard web image defaults to port 8000. `Dockerfile.sagemaker` instead sets port 8080 and `/opt/ml/model`, with `/ping` and `/invocations` aliases. Chapter 37 explains packaging and its remaining cloud verification. [AWS custom inference contract](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html).

The classic endpoint deployment flow creates a SageMaker Model definition linking the image, artifacts and role; an endpoint configuration selecting model variants and capacity; and an endpoint using that configuration. A trusted application calls the SageMaker runtime using AWS-authorised requests. A SageMaker endpoint is not automatically a public anonymous website; a public product needs an application/authentication layer as appropriate.

Local practice should verify input/output contracts, model loading, error behaviour and container startup. SageMaker SDK local mode can exercise supported jobs using Docker, but inspect its configuration because some operations still contact AWS. [AWS local pipeline documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-local-mode.html).

Managed hosting reduces machine-management work but still requires our choices about capacity, security, deployment, metrics, costs and cleanup. EC2 hosting and SageMaker hosting are two learning paths; we do not need both to answer the same production request.

## 24. Continuous delivery, deployment and rollback

**Planned implementation beyond current CI.** CI checks code. Continuous delivery keeps an approved release ready; continuous deployment automatically promotes qualifying changes. Our present workflow stops after testing a temporary container.

The intended extension is: test → build/version image → push ECR → select approved model → deploy staging → smoke/load checks → approval → deploy production → observe → promote or roll back. The application release and model release have separate identities: a code-only fix need not retrain, and a new dataset may produce a model with unchanged API code.

Blue/green deployment runs an old and new environment and switches traffic after checks. A canary sends a small share of traffic to the candidate before wider rollout. Shadow testing sends copies of requests to a candidate without using its answers for business decisions. A/B testing measures outcomes across assigned groups; it needs careful experimental design and enough matured outcomes.

Rollback should restore a previous compatible image, model and schema configuration. It does not undo calls already made or repair corrupted data. If schema compatibility is broken, traffic may need to stop until a safe version is available. Release gates should check error rate, response time and technical correctness immediately; delayed business metrics require a longer observation period.

## 25. Monitoring, data drift and delayed outcomes

**Planned implementation.** Three monitoring layers answer different questions.

| Layer | Observe | Example response |
| --- | --- | --- |
| Service | Request count, error rate, latency percentiles, CPU/memory, restarts | Investigate slow dependencies or add capacity |
| Data | Missing values, ranges, category frequencies, schema failures | Quarantine broken input; investigate upstream changes |
| Model/business | Precision/recall, calibration, top-k yield and outcomes by group | Evaluate degradation and candidate improvements |

Average latency can hide slow requests; p95 is a time below which 95% of observed requests complete. Choose service targets based on user needs and test realistic traffic. Log request ID, timestamp, release/model ID and error category without exposing unnecessary customer details. Do not indiscriminately log bank input payloads. Retention and access rules matter.

Data drift is a change in input distributions. For example, later campaigns might include many more retired customers. Concept drift means the relationship between inputs and outcomes changed: similar customers now respond differently. Changes in prediction-score distribution are another signal. None alone proves the model is wrong or identifies the cause.

A local monitor can compare current numeric summaries, categorical frequencies, missingness and out-of-range rates against a versioned reference window. Statistical measures such as PSI, KS tests or distribution distances can help, but thresholds depend on sample size, binning, seasonality and practical impact. Tiny batches create unstable alerts; large samples can make unimportant differences statistically significant.

Performance monitoring requires actual outcomes joined to predictions. Define a maturity window, measure label coverage and avoid classifying unobserved outcomes as failures. Keep separate reports by model version and relevant population segments so an overall average does not hide local degradation.

The cloud design can capture approved telemetry into S3, run scheduled monitoring jobs, publish metrics and send actionable alarms through the selected monitoring stack. AWS monitoring options have service/account availability constraints; verify current documentation when implementing them rather than assuming a particular managed monitor is available. [AWS monitoring overview](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html).

An alert should include what changed, the affected period/version, evidence and the next investigation step. It should not automatically replace a model because one drift threshold was crossed.

## 26. Retraining and improvement decisions

**Planned implementation.** Retraining may be scheduled, triggered by sufficient new labelled data, or requested after investigation. A drift alert starts an investigation; it is not an automatic instruction to train and deploy.

A controlled workflow checks label maturity and data quality, versions the training snapshot, trains a candidate, compares it fairly, reviews fairness/calibration and business constraints, then approves or rejects it. Keep the existing serving model when the candidate fails. Preserve evidence even for rejected experiments.

Possible improvements include better point-in-time features, better labels, more representative data, regularisation, a different algorithm, calibrated probabilities, a threshold tied to calling capacity, and more efficient batch processing. A more complex model is not automatically an improvement.

| Scenario | Investigate first | Possible action |
| --- | --- | --- |
| Many invalid requests | Input schema and upstream producer change | Fix contract/producer; do not retrain on broken rows |
| More unknown job categories | Category source and population coverage | Update schema/features and evaluate a candidate |
| Drift with stable measured performance | Seasonality, sample size, reference window | Observe with documented justification |
| Performance drops after outcomes mature | Data/label quality, selection bias, changed relationships | Build and compare a candidate |
| New release returns errors | Container/model compatibility and configuration | Roll back the release |
| High scores but few subscriptions | Calibration, policy, outcome window, population shift | Re-evaluate decision policy and model |
| New model is worse | Split validity, features and training behaviour | Reject it; retain the approved model |

Never feed the model's own predictions back as if they were verified customer outcomes. Doing so can reinforce mistakes. Changes to who gets called also change which outcomes we observe; record that selection policy.

## 27. Scaling, reliability and cost decisions

**Conceptual design.** Identify the bottleneck before adding infrastructure.

For online serving, measure concurrent requests, latency, memory and CPU. More workers can increase throughput but also duplicate model memory. Stateless replicas allow a load balancer to distribute requests; request-specific state should not depend on one container's local memory. Autoscaling takes time, so safe minimum capacity and burst behaviour matter. We have not implemented or load-tested this architecture.

For batch scoring, load the model once per process, process chunks, and write partitioned outputs rather than loading every customer into memory at once. Parallel execution needs stable IDs, retry-safe output handling and deterministic release selection. For larger training data, optimise data preparation and consider suitable algorithms before assuming distributed training is necessary.

Set timeouts and bounded retries for appropriate transient failures. Retry only operations that are safe to repeat, with backoff; endless retries can amplify an outage. Define a fallback: a previous approved model, a transparent business rule, or refusing to score until safe. Do not manufacture predictions during an outage.

Cloud costs include more than training time: endpoint/instance runtime, attached disks, snapshots, S3 objects, ECR storage, log retention, public IPs, load balancers and data transfer may all matter. Stopping compute can leave storage and other resources behind. Cleanup must be documented beside resource creation and verified afterwards. Check current region/account-specific prices before actual deployment. [SageMaker pricing](https://aws.amazon.com/sagemaker/ai/pricing/).

## 28. Troubleshooting and recovery

Read the final error line, identify the stage that failed, then check the items below. Do not paste tokens or private customer data into logs or issues.

| Symptom | Likely reason | Fix |
| --- | --- | --- |
| `python: command not found` | Environment not activated or only python3 exists | Run `python3 --version`, create the venv and activate it |
| `No module named bank_ml` | Wrong interpreter or installation missing | Activate `.venv`, run `python -m pip install -e '.[dev]'` from project root |
| Editable install says setup.py is missing | Old pip cannot install this pyproject | Run `python -m pip install --upgrade pip`, then retry |
| `no matches found: .[dev]` | zsh interpreted square brackets | Quote the argument: `'.[dev]'` |
| Dataset download fails | Offline, UCI unavailable, proxy or certificate issue | Check network, retry later; never disable TLS verification |
| Model file not found | Training has not completed or wrong working directory | Run `python -m bank_ml.train` in project root; inspect artifacts |
| API exits before starting | Missing or incompatible trusted model | Read startup error; retrain in the same environment |
| Address already in use | Another process uses port 8000 | Stop the earlier server with Control+C or choose another port |
| HTTP 422 | Input violates required fields, types or ranges | Start from examples/customer.json; read the response detail |
| Docker daemon unavailable | Docker Desktop engine stopped or inaccessible | Start Docker Desktop, wait, then run `docker info` |
| Release check fails at container startup | Missing/modified release or library mismatch | Restore the reviewed release or promote a compatible model; do not bypass checks |
| Git push says authentication failed | GitHub login missing or invalid | Run `gh auth login` interactively; do not put credentials in repository files |
| CI fails in download step | External dataset source unavailable | Read the logs and rerun later; tests themselves are offline |
| CI does not start | Actions disabled or billing allowance exhausted | Check repository Actions settings and account billing |

### Collect useful debugging information

Run from the project root:

```bash
pwd
python --version
python -m pip --version
python -m pytest -q
git status
```

For containers, run `docker ps` to see running containers and `docker logs bank-ml` to read logs while the named container exists. Do not solve a port problem by opening your machine to the internet.

### Rebuild safely

A virtual environment can be recreated from project dependencies. Do not delete project source, datasets or Git history while troubleshooting. Generated models can be recreated by training. A model trained in one library environment may not load in a different one: rebuild it using the intended runtime.

## 29. Exercises and interview checkpoints

| Exercise | Expected lesson |
| --- | --- |
| Send age -1 through `/docs` | HTTP 422; validation happens before prediction |
| Omit `job` | Missing required input is rejected |
| Add `duration` to the API input | Extra fields are rejected by the API |
| Use a new job category | Request is accepted; unknown-category encoding does not imply reliability |
| Start with `MODEL_PATH` pointing at a nonexistent file | Startup fails; health never becomes ready |
| Change a local JSON input | Predictions may change without retraining |
| Train into a different output directory | The default API still uses its default model unless configured |
| Change a model after building an image | Existing image still contains its build-time model |
| Stop the server and request health | Connection fails; no process is listening |
| Read metrics before celebrating a green CI run | Software correctness and useful model performance differ |

Checkpoint answers: S3 would store data, not execute this API. Training fits parameters; inference uses them. The image is a template; the container runs it. Git saves local history; GitHub hosts and automates it. A model file is an artifact, not source code. An AWS budget alert is not a guaranteed spending cap.

For interviews, explain the implemented system honestly, then distinguish the planned architecture. Be able to draw the training and serving flows, explain why duration is excluded, describe delayed-label monitoring, justify batch versus online, and outline how rollback works. Evidence of a tested local container is valuable; it is not evidence that you operated a production AWS endpoint.

## 30. Implementation sequence from here

The remaining implementation should follow this order, with each stage updating this README in place:

1. Versioned data/model manifests and SageMaker-compatible local training/serving adapters.
2. Validated batch scoring with identifiers, model version and retry-safe outputs.
3. S3 interface with local mocks, followed by optional real integration verification.
4. Candidate comparison, approval states and local release/rollback practice.
5. Prediction telemetry, delayed-label joins, data-quality/drift reports and performance checks.
6. Controlled candidate retraining and failure scenarios.
7. AWS infrastructure and deployment definitions for separate EC2 and SageMaker exercises.
8. Optional bounded cloud execution when budget/sandbox access permits, including load checks and verified teardown.

Each stage needs executable code, meaningful checks, expected results, failure exercises and explicit verification evidence. A future lesson should never silently turn a planned feature into a claimed deployment.

## 31. Validation evidence and measured limitations

### Local checks, 6 November 2026

- Python 3.10.0 on macOS; scikit-learn 1.7.2.
- Installed the project in an isolated virtual environment.
- Downloaded UCI bank-full.csv and recorded its SHA-256 and license metadata.
- Four pytest tests passed, covering training/save/load, prediction, API validation, leakage exclusion, unknown categories, and missing-model startup.
- One upstream TestClient deprecation warning was emitted; tests passed. This is a dependency migration notice, not a failed API request.
- Trained on 36,168 earlier rows and tested on 9,043 later rows.
- Saved measured results in [baseline-metrics.json](docs/baseline-metrics.json).
- CLI prediction returned approximately 0.04624 for the fictional example customer.

### How to interpret the first results

Average precision was 0.5501 versus a holdout subscriber rate of 0.3158. ROC AUC was 0.7173. At a threshold of 0.5, precision was 0.7042 but recall was only 0.0525: the model missed most subscribers at that threshold. We have not selected a useful operating threshold.

The earlier training period's subscriber rate was 0.0673, while the later holdout's rate was 0.3158. This large difference illustrates why future-looking evaluation matters. Campaign selection and population changes can affect results. We cannot infer the cause of this difference from these aggregate numbers alone. An apparently good ranking score does not establish well-calibrated probabilities or business suitability.

### Reproduce

Follow README commands from a fresh environment. Model/data files are generated locally, not committed. Dependencies are specified as ranges, so exact bytes or metrics may vary with future versions. Data SHA-256 and library versions are included in the report.

### Container verification

Built `bank-ml:local` with Docker Desktop 29.8.0. Started a temporary non-root container. GET `/health` returned `{"status":"ready"}`; POST `/predict` returned approximately 0.04624 for the example customer. The temporary container was removed after verification; the reusable image remains. The image uses Python 3.11, while the original laptop training used Python 3.10. The tested serialization loaded successfully, but cross-version compatibility should not be assumed for other builds.

### Hosted CI

Initial run: https://github.com/subhashdixit/bank-customer-subscription-ml-pipeline/actions/runs/42107025182

Result: **passed**, 1 minute 22 seconds. GitHub Linux/Python 3.11 installed dependencies, passed pytest, downloaded real data, trained the model, predicted the example, built Docker and checked the running container. This verifies a second environment in addition to the laptop.

GitHub emitted informational notices about Node.js action-runtime migration and the upcoming `ubuntu-latest` image change. The current run passed; future maintenance should update the action versions and choose an explicit runner image when strict reproducibility is needed.

The documentation-only follow-up commit uses `[skip ci]` to avoid repeating compute for an evidence update. The linked run validates the application and workflow commit.

No AWS resources have been created and no real cloud deployment has been verified.

### Full pipeline and documentation review, 6 November 2026

Reviewed all application modules, tests, Dockerfile, workflow and learning documents together. Fixed stale private-repository descriptions. Added the main study guide with an architecture diagram, input contract, module walkthrough, workflow explanation, metric examples, exercises and production limitations. Checked that local Markdown links resolve.

Repeated local pytest (4 passed), real-data training and CLI prediction. Results matched the earlier baseline. The existing upstream local TestClient deprecation warning remains informational.

Strengthened the actual container smoke check: health JSON, finite prediction in [0, 1], numerical parity with the saved host model, and HTTP 422 for invalid age. Updated checkout/setup-python to Node 24 action versions and selected Ubuntu 24.04 explicitly.

Updated code/workflow commit: `9e00682`.

Hosted verification: https://github.com/subhashdixit/bank-customer-subscription-ml-pipeline/actions/runs/42108135672

Result: **passed**, 48 seconds. Dependency install, tests, public download, training, CLI prediction, Docker build and strengthened HTTP checks all passed. The earlier action-runtime and runner-migration annotations were not emitted in this run. The follow-up evidence-only commit skips CI because no executable files change.

This verifies the currently implemented local/CI pipeline. AWS, persistent deployment, model registry, quality gates, monitoring and retraining remain planned rather than verified features.

### Single-document consolidation, 6 November 2026

Consolidated the learning material into this README and removed the seven separate Markdown guides. Added a single chapter sequence and same-page contents, preserving attribution, measured results, commands and limitations. Expanded conceptual coverage for batch data, versioning, EC2, SageMaker, delivery, monitoring, drift, retraining and scaling. These additions describe planned components and do not claim new executable features. Checked Markdown structure, local file links and internal contents anchors. This documentation-only change does not alter the previously verified application or workflow.

### Integrated retrieval verification, 8 November 2026

Code commit: `dd05d49`. Local checks passed: eight pytest tests, six retrieval regression cases and `pip check`. Built the updated Docker image and verified real HTTP health, numerical prediction, invalid-input rejection and `/predict/briefing` score parity/product-scoped citations. Removed the temporary container after verification. The example customer retained probability approximately 0.046237.

Optional provider success, invalid output, forged IDs and timeout behaviour were tested with HTTPX simulated responses. No live LLM provider call was made, no credentials were committed, and no AWS resources were created. The pre-existing local TestClient deprecation warning remains informational.

Hosted verification run: https://github.com/subhashdixit/bank-customer-subscription-ml-pipeline/actions/runs/42410350256

### Final integrated pipeline verification, 8 November 2026

Final executable revision: `c33ac18`. Hosted run [42410671032](https://github.com/subhashdixit/bank-customer-subscription-ml-pipeline/actions/runs/42410671032) **passed in 1 minute 11 seconds**: eight tests, six retrieval cases, synthetic software-check training, CLI prediction, Docker build, and real HTTP checks including the enriched briefing. The optional real-data job was deliberately skipped on this push. Initial run 42410350256 failed at the UCI download timeout; it is not a successful full run. The README-only evidence update skips CI.

This final evidence verifies the integrated ML/retrieval software and simulated LLM adapter, not a live LLM provider, model-quality improvement or AWS deployment. Chapter 33 explains how to enable the optional provider safely.

## 32. Glossary

| Term | Meaning here |
| --- | --- |
| Repository | Project files and their Git history |
| Commit | A saved snapshot of tracked changes |
| Dependency | Another library our application needs |
| Virtual environment | An isolated collection of Python packages |
| Module | A Python file that can be imported or executed |
| Feature | An input used by the model |
| Label / target | The historical answer the model learns to predict |
| Training | Finding model parameters from examples |
| Inference | Using a trained model to predict |
| Artifact | A generated output, such as a model or report |
| Holdout | Data kept out of training for evaluation |
| Leakage | Information used in evaluation that would not be available at prediction time |
| Preprocessing | Converting raw inputs into suitable model inputs |
| API | An agreed interface between programs |
| HTTP | The request/response protocol used by our web API |
| Endpoint | A specific API route; AWS also uses the term for a hosted model service |
| JSON | A text format for structured values |
| Port | A numbered network listening location |
| Image | A packaged application template for containers |
| Container | A running instance of an image |
| CI | Automated checks on code changes |
| Continuous delivery | Keeping a checked release ready to deploy |
| Continuous deployment | Automatically deploying changes that pass release checks |
| Pipeline | A sequence of connected steps |
| Rollback | Returning to an earlier known release |
| S3 bucket / object | A storage container / a named stored item |
| IAM role | An assumable AWS identity with permissions |
| Region | A geographic AWS service location |
| Infrastructure as code | Files describing resources to create and configure |
| Drift | A change in data or relationships that may affect a model |
| OIDC | Identity federation used to obtain temporary credentials without storing AWS keys in GitHub |

## 33. Integrated product retrieval and optional LLM briefing

### What has changed in the same project?

The existing customer subscription model remains the source of the numerical score. A new `/predict/briefing` route accepts a customer, a selected demo product and a question. It returns that same score plus cited product information and follow-up questions. This is not a separate chatbot or a new predictive model.

```mermaid
flowchart TD
    I[Customer + product ID + question] --> M[Original classifier]
    I --> F[Select only this product's document chunks]
    F --> V[Local TF-IDF vectors and FAISS retrieval]
    V --> L{Request LLM and provider configured?}
    L -->|Yes| C[LLM selects allowed evidence and question IDs]
    L -->|No or failed| E[Deterministic extractive briefing]
    C --> S[Validate selections; reject unknown IDs]
    S --> R[Code renders source text and citations]
    E --> R
    M --> O[Response: original score + grounded briefing]
    R --> O
```

The classifier is **not product-aware**: changing the selected product changes retrieved information, not its probability. Our historical training data has no matching product documents, so this integration cannot establish increased prediction accuracy. It improves the information delivered alongside the prediction.

### The fictional knowledge collection

`src/bank_ml/knowledge/deposits.json` contains two fictional demonstration products. Twelve-month terms specify INR 10,000 as the minimum; six-month terms specify INR 5,000. Each product has four short sections: minimum deposit, duration, withdrawal and eligibility. Interest rates are intentionally absent. These are project-authored learning examples, not real bank policies.

Each section is one chunk with a stable ID. A tiny source already divided into short sections does not need an arbitrary overlapping chunk splitter. For longer documents, preserve section/source metadata while splitting, and evaluate how chunk size affects retrieval.

The JSON is included as Python package data, so it is copied into the installed Docker application. `knowledge_version` identifies the authored collection, while `knowledge_sha256` fingerprints its exact bytes. Restart/rebuild after editing it. Old process memory and old images do not automatically acquire the new terms.

### What FAISS does here

`ProductKnowledge` fits one TF-IDF vectorizer and builds one FAISS `IndexFlatIP` per product at process startup. TF-IDF converts each section heading and its text into weighted word/bigram vectors; it requires no external model download or embedding API. We normalise vectors and use inner products as cosine similarity. [FAISS distance documentation](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances).

This is a **lexical retrieval baseline**, not neural semantic embeddings. Synonyms can be missed, and shared words can retrieve weak evidence. Only positive-overlap passages are returned, up to four; a zero-vector query returns no evidence. Positive overlap is not a calibrated confidence score. A production abstention policy needs an evaluated threshold and broader queries. Neural sentence embeddings are a future improvement requiring retrieval comparison, pinned model versions and corresponding index rebuilds.

Filtering happens before search by selecting a product-specific index, so another product's minimum cannot enter the candidate set. Indexes live in memory and are rebuilt from the small versioned JSON on startup. No separately managed vector database or persisted index service is required here. This is appropriate for eight chunks, not a claim about large-corpus scalability.

### Run the enriched prediction without any key

Use the earlier environment/training setup, then reinstall updated dependencies:

```bash
python -m pip install -e '.[dev]'
python -m uvicorn bank_ml.api:app --host 127.0.0.1 --port 8000
```

In another terminal at the repository root:

```bash
curl http://127.0.0.1:8000/products
curl -H 'Content-Type: application/json' --data @examples/briefing.json http://127.0.0.1:8000/predict/briefing
```

You can also use `/docs` and paste `examples/briefing.json` into POST `/predict/briefing`. `customer` contains the same ten fields used by the original endpoint. `product_id` chooses `demo-12-month` or `demo-6-month`; `question` controls retrieval; `use_llm` defaults to false. Unknown products return HTTP 404 when knowledge is available. Invalid customer fields return HTTP 422 before any LLM call.

The response includes:

| Field | Meaning |
| --- | --- |
| `subscription_probability` | Unmodified classifier result |
| `briefing.product` | Selected fictional product |
| `briefing.mode` | `extractive`, `llm_curated`, `no_evidence` or `unavailable` |
| `briefing.llm_status` | Why generation was used, skipped or unavailable |
| `briefing.product_information` | Exact excerpts linked by `source_id` |
| `briefing.sources` | Retrieved text, section, document, version and similarity |
| `briefing.questions_to_confirm` | Fixed safe follow-up questions |
| `briefing.knowledge_version` and `knowledge_sha256` | Collection provenance |
| `briefing.limitations` | Product, currency and model limitations |

No key is required for `extractive` mode. This mode is retrieval plus deterministic rendering, **not an LLM-generated answer**. It preserves the useful part of the response when a provider is unavailable.

### Example from the verified local container

For the committed `examples/briefing.json`, the unchanged model returned approximately **0.046237**, or **4.62%**. The response included these excerpts with source IDs:

- “The minimum opening deposit is INR 10,000.” — `demo12-minimum`
- “The deposit term is 12 months.” — `demo12-duration`
- “Early withdrawal is permitted with a penalty. The penalty amount is not specified in this demonstration document.” — `demo12-withdrawal`

The response also includes eligibility limitations and questions about intended amount, access to funds and separately checked eligibility. These are fictional product terms, not information about a real offer. The earlier conversational 18% example was illustrative; it is not this customer's measured model result.

### Optional LLM setup

The provider adapter expects an OpenAI-compatible `POST /chat/completions` API, not any arbitrary provider protocol. The base URL, model name and key must match your provider's documentation. No provider is assumed or preselected, and no key has been supplied or committed.

Set variables in the terminal where you start Uvicorn:

```bash
export LLM_BASE_URL='https://your-provider.example/v1'
export LLM_MODEL='your-model-name'
read -s LLM_API_KEY
export LLM_API_KEY
```

The `read -s` command waits for you to paste the key without echoing it. Restart Uvicorn in that environment. Change `use_llm` to true in a local copy of the example request. `.env.example` lists placeholders only; this application **does not automatically load `.env` files**. Docker users can explicitly pass a local ignored env file with `docker run --env-file .env ...`; never copy it into an image or Git.

When requested and configured, the adapter sends only the question and retrieved demonstration text to the provider. It does not send customer features, balance or score. Do not place private customer information in the question. Provider use may incur charges. Requests have a 15-second HTTP timeout, no redirects, no automatic retry, and a small output-token limit. This is a bounded demo, not a rate-limited public service. The URL is server configuration, not user input; HTTPS is required except for local development hosts.

### Why generated facts cannot replace source facts

The LLM is constrained to select source IDs and follow-up question IDs. Code rejects unknown IDs, malformed JSON, extra fields and empty evidence selections. It then renders exact text from the retrieved sources. No free-form LLM financial advice or generated numeric score is shown.

This is **constrained retrieval-augmented generation/curation**, rather than an unrestricted conversational assistant. It teaches the integration while reducing unsupported claims. Documents and questions are treated as untrusted content, not instructions. Allowlisting output IDs prevents a prompt from injecting a fabricated interest rate or another product's terms into the rendered answer. It does not guarantee the LLM always chooses the most relevant valid passage; relevance still needs evaluation.

`llm_status=not_configured` means one or more variables are missing. Network failure, provider errors, unsupported protocol or invalid output trigger extractive fallback. Logs do not print credentials or provider payloads. If knowledge loading or briefing construction fails, the enriched route preserves the numerical score with an unavailable notice. Original `/predict` remains independent of provider availability.

### Important interpretation limits

The historical `balance` field is in euros. Demo terms are in INR. We deliberately make no comparison between the two, infer no affordability and perform no currency conversion. Subscription likelihood is not eligibility or suitability. Retrieved terms cannot explain why the classifier produced its score. Model-specific attribution is a separate future implementation.

Unknown or unsupported questions may yield no evidence, but lexical overlap can still produce irrelevant excerpts. For example, asking an interest-rate question containing the word “deposit” may retrieve deposit passages without a rate. The application quotes those passages rather than inventing a rate; the user must see that the requested fact is absent. A larger evaluation set and better abstention remain necessary.

### Evaluation and CI

```bash
python -m pytest -q
python scripts/evaluate_retrieval.py
```

The new tests exercise product isolation, unknown questions, exact-source rendering, validated provider selections, forged source IDs, empty evidence, provider timeout/missing credentials, unchanged scores and knowledge failure isolation. Provider behaviour is simulated using HTTPX mock transport; no paid call is made by tests.

Six handwritten retrieval checks cover both products, expected top matches and a no-overlap query. They are a regression fixture, not a general benchmark or proof of semantic understanding. Expand it with paraphrases, misleading overlaps, missing facts and adversarial questions before claiming broader retrieval quality.

GitHub CI installs FAISS, runs tests and retrieval checks, trains a synthetic smoke-test model, builds the container and sends real HTTP requests to both prediction endpoints. `scripts/check_api.py` verifies the same numerical score and product-scoped citations in the running container. No LLM secret is needed in CI. Live provider compatibility cannot be verified until a provider is configured; simulated adapter tests are labelled as such.

For future monitoring, track retrieval hit/miss rates, question classes, response modes, provider timeouts, latency and cost, separately from predictive model metrics. Product-document updates require retrieval evaluation; they do not require classifier retraining. Keep the document hash, prompt/adapter version and provider model alongside release evidence. Public hosting, authentication, rate limits, neural embeddings and actual AWS deployment remain separate future work.

### Real-data training versus routine CI

The first RAG run passed tests/retrieval but failed on a 120-second UCI connection timeout. To keep software verification independent of that external source, routine CI now generates 300 deterministic synthetic rows using `scripts/make_smoke_data.py`. Its fitted model exists only for container/API checks, is not published as a release, and its metrics must not be presented as model-quality evidence. Local `python -m bank_ml.train` still uses real `data/bank-full.csv` by default and remains the source of the documented bank baseline metrics.

To request real-data hosted verification, open Actions → Test and build → Run workflow and enable `real_data`. The additional job downloads UCI data and trains/evaluates it, and can fail independently if UCI is unavailable. Previous real-data evidence remains historical; it is not relabelled as a new successful run. Dependency installation still requires package services and Docker image downloads; this change removes the live dataset dependency from normal checks, not every network dependency.

## 34. Flask web app and free Render deployment

### One service, not two servers

Flask renders the customer form and serves CSS/JavaScript. FastAPI keeps the existing validated prediction routes. `a2wsgi.WSGIMiddleware` mounts Flask at `/` **after** the API routes, so `/docs`, `/predict`, `/products` and `/predict/briefing` remain reachable. Uvicorn serves the combined application as one process on one port. There is no Flask development server, browser-to-private-API URL, CORS workaround, Node build or second paid instance. [FastAPI WSGI integration](https://fastapi.tiangolo.com/advanced/wsgi/).

The browser sends JSON to its own origin, displays the score and exact retrieved excerpts, and exposes source IDs and limitations. Text is inserted using `textContent`, not untrusted HTML. Flask adds a restrictive content security policy to the page/assets. Browser validation helps usability; server validation remains authoritative. No customer records are saved by the application. Hosting access logs and providers have their own retention; use fictional inputs only.

The public form always uses extractive mode. `ALLOW_LLM_REQUESTS=false` also blocks attempts to enable provider calls through the API, preventing visitors from spending a configured provider key. This is a public learning demo without user accounts, not a bank production portal. Real customer data, private documents, provider calls or production use require authentication, request limits and an appropriate data-handling design.

### Run the exact release locally

```bash
python -m pip install -e '.[dev]'
PORT=8001 python -m bank_ml.serve
```

Open http://127.0.0.1:8001/ for the form; `/docs` remains available for developers. If that port is in use, choose another free port. The entry point binds on all interfaces to support containers; when running directly, use trusted local networks/firewall settings. For a localhost-only development process use `MODEL_PATH=release/model.joblib python -m uvicorn bank_ml.api:app --host 127.0.0.1 --port 8001`.

The form starts with the existing fictional customer, whose released model estimate is approximately 4.62%. Select the six-month product and run again: product terms change while the same customer's score stays unchanged. Try an unrelated question: no matching evidence should be reported for a no-overlap query. Reset restores the example. Loading and error states keep the interface usable when requests fail.

### The release boundary

`artifacts/` is for experiments. `release/` is a small, intentionally committed snapshot of the trusted UCI-trained baseline, containing a model, metrics and manifest. We commit this small snapshot to make free hosting reproducible without a dataset download on every build. This is an educational convenience; a larger system would use a versioned artifact store/registry and approvals instead of committing binary models.

`bank_ml.serve` checks model/report hashes and exact scikit-learn version **before** loading the model. Hashes detect a mismatch against the trusted manifest; they are not signatures or proof that an unknown model is safe. Never substitute untrusted joblib files. The original release was trained on Python 3.10 and is container-tested on Python 3.11; reproducible future releases should train in the deployment environment. Scikit-learn is pinned at 1.7.2; remaining dependencies/base image are not yet fully locked.

To promote a new trusted run after review:

```bash
python -m bank_ml.train
python scripts/promote_model.py --expected-data-sha256 d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d
```

This command checks that metrics refer to the explicitly expected dataset and copies the model/report into `release/`. It does not decide model quality for you. If the intended dataset changes, independently verify its provenance and supply its actual approved checksum, rather than blindly reusing this example. The script currently uses the baseline release label; update it for a genuinely new release. Review the diff, evaluation and manifest before committing. Never promote the synthetic CI model.

### Render configuration, explained

| Setting | Value | Reason |
| --- | --- | --- |
| Runtime | Docker | Use the tested Python environment |
| Plan | Free | User's zero-paid-hosting requirement |
| Region | Oregon | Matches the selected Render service; AWS lab region is independent |
| Branch | main | Deployment source |
| Dockerfile | `./Dockerfile` | Includes Flask, model and knowledge |
| Port | Hosting `PORT`, example 10000 | Must listen on `0.0.0.0`, not localhost |
| Health path | `/health` | Startup must load the model before readiness |
| Auto deploy | `checksPass` in `render.yaml` | Build after linked CI succeeds |
| LLM calls | Disabled | No API key or provider spend for public visitors |

`render.yaml` is infrastructure configuration, not a live service by itself. Blueprint deployments read it. A manually created web service must be configured equivalently; changing YAML does not automatically change a service that was never linked to a Blueprint. [Render Blueprint specification](https://render.com/docs/blueprint-spec), [Docker deployment](https://render.com/docs/docker).

Deployment sequence:

1. Sign in to Render, use the existing GitHub connection and select this repository.
2. Create a Web Service using Docker, branch main and the Free instance type. Do not select a paid tier or persistent disk.
3. Set health path `/health`; configure auto deploy after CI passes if using the connected Git provider.
4. Keep LLM calls disabled; there are no secrets needed for this demonstration.
5. Render clones the commit, builds the image and starts `python -m bank_ml.serve`. It verifies the release, loads the model and builds the tiny FAISS indexes.
6. After health succeeds, Render routes HTTPS traffic to the process and assigns an `onrender.com` URL. Test the page, predictions and product switching at that actual URL.
7. On later updates, observe both CI and Render deployment status. A GitHub push alone is not proof that the new service is healthy.

Free services sleep after 15 minutes without traffic and can take around a minute to wake. The filesystem is ephemeral; this app rebuilds its small index and uses bundled release files rather than storing user uploads or mutable databases. The workspace's free hours/build/bandwidth allowances are shared with other services; check total usage, especially if the account hosts several demos. No keep-alive workaround is used. [Render free limitations](https://render.com/docs/free).

### What Render teaches versus AWS

Render handles VM allocation, process supervision, HTTPS routing and much of deployment orchestration. The model/container/API responsibilities are ours. On EC2 we must explicitly manage those hosting responsibilities. The following labs use the **same release and code**, so Render is a free demonstration venue, not a replacement for learning AWS.

## 35. AWS preparation, identity, storage and container registry

**Future billable lab. No AWS resources have been created or verified here.** These are project-specific exercises, not a claim to cover every AWS service. Before executing, check current prices, quotas and the account/region. You do not need to run them to use the Render demo.

### Choose a path before creating resources

- **EC2 path:** deploy the complete Flask + API + FAISS application with `infra/aws/ec2.yaml`.
- **SageMaker path:** practise managed numeric inference separately, using `Dockerfile.sagemaker`. A full web UI still needs web hosting. Managed training uses another image, `Dockerfile.training`.
- Do not provision both paths, a NAT gateway, load balancer, database and GPU merely because they exist. Each resource must have a purpose and a cost estimate.

### Account and identity checklist

Use MFA and an appropriate developer/SSO role instead of using the account root user for routine work. Install AWS CLI v2 from AWS's official instructions, configure a supported login/profile and verify the current identity:

```bash
aws sts get-caller-identity
aws configure get region
```

The account ID is not a credential. Access keys, session tokens and provider keys are credentials and must stay out of Git and images. CLI authentication and browser-console authentication are separate sessions. A valid identity still needs permission for the requested resource operations.

Set the chosen region explicitly in lab commands. The examples use Singapore (`ap-southeast-1`); this is not a promise about the cheapest region or instance availability. ECR images, S3 data, training jobs and endpoints should normally stay in the intended region. Cross-region movement adds latency/cost and can cause access errors.

Four identities have different responsibilities:

| Identity | Responsibility | Example permission boundary |
| --- | --- | --- |
| Developer/provisioner | Create approved infrastructure | Scoped CloudFormation, EC2, IAM role creation/pass-role, ECR and S3 operations |
| CI deployment role | Publish a checked image/release | Only the chosen ECR repository and deployment targets |
| EC2 instance role | Pull/run the app, use SSM | ECR pull for one repository, SSM management |
| SageMaker execution role | Run a managed job or load its artifacts | Read specified S3 data/model prefixes and ECR image; write chosen output/log locations |

`iam:PassRole` allows passing a role to a service; it is not equivalent to permission to assume any role. An IAM **trust policy** names who may assume the role; a **permission policy** names what that assumed role can do. Missing either can break deployment. Use scoped resource ARNs and service conditions rather than granting administrator access to fix an error.

### Storage and versioning

S3 stores objects under keys, not a normal POSIX filesystem. Proposed prefixes are `raw/<dataset-version>/`, `prepared/<version>/`, `models/<release>/`, `reports/<run>/` and `monitoring/<date>/`. Block public access, use encryption, enable versioning where appropriate and define retention/lifecycle rules. A public web app does not imply its data bucket should be public.

Create a private, globally unique bucket in the chosen region using the console, and record its name. Upload only the required public learning data or reviewed artifacts. Do not upload real customer records for this exercise. An IAM policy can permit `GetObject` for a model prefix without permitting deletion of the entire bucket. SSE-KMS adds key-policy/permission considerations; SSE-S3 avoids requiring a custom key for this introductory exercise.

### Publish the image to ECR

ECR stores the container image; S3 stores datasets/model archives. A container image tag can point to different bytes unless immutability is enforced. An image digest identifies the actual image content.

The following commands create a billable repository and publish an image **only when you run them**:

```bash
aws ecr create-repository --region ap-southeast-1 --repository-name bank-subscription --image-tag-mutability IMMUTABLE --image-scanning-configuration scanOnPush=true
bash scripts/push_ecr.sh ap-southeast-1 bank-subscription release-v1
```

The helper uses the current CLI identity, obtains an ECR login token through `--password-stdin`, builds for **linux/amd64**, pushes and prints the digest URI. The x86 EC2 and example SageMaker machines require this architecture; an image built by default on an Apple Silicon laptop may be ARM instead. Docker Buildx can emulate another architecture, which may build more slowly. Reusing an immutable tag with changed bytes fails intentionally: use a new release tag.

Keep the printed URI, for example `ACCOUNT.dkr.ecr.ap-southeast-1.amazonaws.com/bank-subscription@sha256:DIGEST`. ECR login authenticates Docker to that registry; it does not grant permissions absent from IAM. A scan report needs review; enabling scanning is not a security approval by itself. [ECR lifecycle tutorial](https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html).

## 36. AWS EC2 deployment lab

### What the included template creates

`infra/aws/ec2.yaml` creates a security group, narrowly scoped ECR pull role, SSM-enabled instance profile and a single Amazon Linux 2023 x86 instance with encrypted 20 GB gp3 root storage. It requires an existing VPC and public subnet. It opens port 80 only to the CIDR you provide and does not open SSH. IMDSv2 is required. A public IPv4 and the VM/disk are chargeable resources.

A **VPC** is the network boundary; a **subnet** is a segment in that network; a route to an **internet gateway** permits the public-subnet path. A public IP alone is insufficient without routing. Security groups filter traffic at the resource, while IAM controls service actions. Default egress permits outbound connections needed for package downloads, ECR and SSM. Private-subnet designs need suitable endpoints or NAT routing and different cost planning; this template does not create those.

The instance's startup script installs Docker, authenticates to ECR using its role, pulls the selected image and installs a systemd service. Systemd restarts the container after a crash/reboot, and Docker log rotation limits disk growth. This is a single-instance learning setup with a single point of failure, not a highly available production architecture.

### Before execution

1. Choose a public subnet **inside the same VPC** and verify the route table has a route to an attached internet gateway.
2. Verify account permissions, instance quota, current instance pricing and availability. T3 burst-credit settings can add charges under sustained CPU use.
3. Note your public IPv4 as `YOUR_IP/32`. Do not use a private `192.168...` address or automatically open the app to everyone.
4. Publish the amd64 image and copy its ECR digest URI.
5. Inspect the template, particularly role permissions, ingress and user data. It is locally reviewed but not executed on AWS.

Syntax/service validation and actual creation are different:

```bash
aws cloudformation validate-template --region ap-southeast-1 --template-body file://infra/aws/ec2.yaml
```

This command checks AWS template acceptance, not the correctness of package installation, permissions, subnet routing or app startup. It requires AWS access and has not been run here.

When budget is available, replace every placeholder and execute:

```bash
aws cloudformation deploy --region ap-southeast-1 --stack-name bank-subscription-lab --template-file infra/aws/ec2.yaml --capabilities CAPABILITY_IAM --parameter-overrides VpcId=vpc-REPLACE PublicSubnetId=subnet-REPLACE AllowedCidr=YOUR_PUBLIC_IP/32 ImageUri=YOUR_ECR_DIGEST_URI RepositoryName=bank-subscription
aws cloudformation describe-stacks --region ap-southeast-1 --stack-name bank-subscription-lab --query 'Stacks[0].Outputs'
```

`CAPABILITY_IAM` acknowledges that the stack creates IAM resources. Outputs include the instance ID and temporary demo URL. CloudFormation reaching CREATE_COMPLETE means resource provisioning completed; this template does **not** use a signal that waits for the app to finish installation. Test application health separately after boot.

The initial URL is restricted **HTTP for a fictional-data exercise**. Do not send credentials or private customer information over it. For a public portfolio URL, add HTTPS before broadening ingress: a domain with DNS pointing to the server, a maintained reverse proxy such as Caddy/Nginx with a valid certificate, app binding to localhost behind that proxy, and appropriate 80/443 rules. Alternatively use an ALB with ACM, accepting its separate cost. The included template does not provision TLS, a domain or a load balancer.

### Verify and troubleshoot

From an allowed client, open `/`, `/health`, `/docs` and run a fictional prediction. Use Session Manager to connect without an inbound SSH port:

```bash
aws ssm start-session --region ap-southeast-1 --target i-REPLACE
```

Inside the instance:

```bash
sudo systemctl status bank-subscription
sudo docker logs bank-subscription
sudo journalctl -u bank-subscription --no-pager -n 50
sudo tail -n 60 /var/log/cloud-init-output.log
```

Interpret failures by layer:

| Symptom | Check |
| --- | --- |
| Stack fails | CloudFormation Events, provisioner permissions, subnet/VPC match, quotas |
| SSM cannot connect | Instance role, SSM agent, outbound HTTPS/routing and session permissions |
| Image pull denied | ECR repository/region, digest, role scope and registry login |
| Container exits | Release checksums, scikit-learn version and model startup logs |
| Local health works but external access fails | Public IP, route table, security group and your current `/32` |
| HTML loads but prediction fails | Browser/API request status, server validation and application logs |
| App is slow | CPU credits, memory, concurrent traffic and worker configuration |

### Update and roll back deliberately

Do not assume that editing the template's ImageUri reruns user data on an existing instance. EC2 user data normally runs on first boot. For this simple lab, connect with SSM, authenticate/pull the new approved image, stop the service, edit the systemd `ExecStart` image URI to the new digest, reload systemd and start it, then verify the app. Save the previous digest first so the same procedure can restore it. This causes downtime; production deployment should use replaceable instances or blue/green rollout rather than ad hoc mutation.

A new image should include the selected model release, not an unreviewed latest experiment. A rollback must restore compatible code, preprocessing, model and schema—not merely an old filename.

## 37. SageMaker training and inference lab

**Optional billable managed-ML exercise.** Unlike EC2, SageMaker allocates and supervises job/endpoint compute. The developer still owns model logic, data/role permissions, deployment choices, metrics and cost controls. The UI is a web application, not something the SageMaker runtime automatically exposes as a normal website.

### First understand the two different container contracts

| Training | Inference |
| --- | --- |
| A finite job reads a channel such as `/opt/ml/input/data/train/` | A long-running process listens on port 8080 |
| Produces model files under `/opt/ml/model` | Loads extracted model files from `/opt/ml/model` |
| Compute ends after the job finishes | Endpoint compute remains active until deleted/scaled as supported |
| Uses `Dockerfile.training` | Uses `Dockerfile.sagemaker` |

Our wrappers reuse the existing application image. The training image runs the original training function against `bank-full.csv`; it does not serve HTTP. The inference image sets port 8080 and a release directory under `/opt/ml/model`, with an ENTRYPOINT that tolerates SageMaker's appended `serve` argument. `/ping` returns readiness and `/invocations` accepts the **same single-customer JSON** as `/predict`. It returns numeric prediction only. The HTTP aliases are covered by local tests; AWS-managed behaviour is not yet verified. [AWS custom inference contract](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html).

### Local contract rehearsal

Build the normal image first, then the wrapper. From the repository root:

```bash
docker build -t bank-ml:web .
docker build -f Dockerfile.sagemaker --build-arg APP_IMAGE=bank-ml:web -t bank-ml:sagemaker .
docker run --rm -p 127.0.0.1:8080:8080 -v "$PWD/release:/opt/ml/model:ro" bank-ml:sagemaker serve
```

In another terminal:

```bash
curl http://127.0.0.1:8080/ping
curl -H 'Content-Type: application/json' --data @examples/customer.json http://127.0.0.1:8080/invocations
```

This checks the local contract, not IAM, model download or the AWS endpoint lifecycle. If port 8080 is already occupied, map a different laptop port while keeping container port 8080. Standard web deployment remains the Render/EC2 image; do not replace it with a managed endpoint and expect a public browser form.

For local training-contract practice, the raw CSV must already be downloaded. Mount its directory read-only as `/opt/ml/input/data/train` and a separate writable output directory as `/opt/ml/model`. Build `Dockerfile.training` with the same base image, then run it. Both chronological partitions still need both target classes. This checks file paths and training execution without allocating AWS resources.

### Managed training sequence

1. Upload the public `bank-full.csv` to an approved private S3 prefix in the target region.
2. Build/push the **training** image to ECR for linux/amd64, using a distinct immutable tag.
3. Create a SageMaker execution role trusted by `sagemaker.amazonaws.com`, with only the relevant S3 read/write prefixes, ECR pull and log permissions. Use an appropriate pass-role condition for the provisioner.
4. Submit a training job selecting File input mode for the `train` channel, the S3 input prefix, output prefix, instance type/count and a bounded maximum runtime. Custom command-line hyperparameters are not yet wired into our wrapper; it runs the fixed baseline.
5. Observe job status/logs. Failed jobs are investigated before retrying. Read the output artifact location from the job description after success.
6. Download the candidate artifact to a trusted review location, evaluate it and promote it with a release manifest before deployment. The training output has model/metrics but no automatic approval manifest; it is not an immediately approved endpoint release.

Managed training job submission is a documented next integration, not an automated script in this repository. The wrapper exists; account-side permissions, job submission and output validation remain real AWS exercises.

### Prepare inference artifacts and deploy an endpoint

The following deploy path uses the already reviewed baseline, so no managed training job is required just to learn inference hosting.

1. Build and push `Dockerfile.sagemaker` to ECR as a separate inference image. Its APP_IMAGE can be the normal local image; push the resulting wrapper image to the chosen repository/tag. Ensure linux/amd64 is built on Apple Silicon.
2. Create a gzip tar with files at the archive root, not nested under `release/`:

```bash
tar -czf /tmp/bank-model.tar.gz -C release model.joblib metrics.json manifest.json
aws s3 cp /tmp/bank-model.tar.gz s3://YOUR_PRIVATE_BUCKET/models/release-v1/model.tar.gz --region ap-southeast-1
```

3. Create/review a SageMaker execution role with trust for SageMaker and read access to that artifact and inference image; grant required log actions. Verify the model/image/role are in the expected account/region. For encrypted customer-managed keys, additional KMS grants may be required.
4. Install the optional SDK locally and run the included explicit deployment exercise. It creates a **Model**, an **EndpointConfig**, and an **Endpoint**, then waits for readiness:

```bash
python -m pip install -e '.[aws]'
python infra/aws/sagemaker.py deploy --region ap-southeast-1 --name bank-lab-v1 --role-arn YOUR_EXECUTION_ROLE_ARN --image-uri YOUR_INFERENCE_IMAGE_URI --artifact-uri s3://YOUR_PRIVATE_BUCKET/models/release-v1/model.tar.gz
```

The example defaults to one `ml.m5.large`; confirm current regional price and quota first. Failed/partial deployment does not silently delete resources. Record the names printed by the script and inspect SageMaker/CloudWatch. These operations can create continuing charges even when the browser is closed.

5. Invoke the runtime, using CLI v2 binary-body handling:

```bash
aws sagemaker-runtime invoke-endpoint --region ap-southeast-1 --endpoint-name bank-lab-v1-endpoint --content-type application/json --body fileb://examples/customer.json /tmp/bank-response.json
cat /tmp/bank-response.json
```

The caller needs `sagemaker:InvokeEndpoint` for this endpoint. The runtime request is AWS-authorised; it is not an anonymous public URL. A future web deployment can invoke this runtime from its server-side role instead of loading the classifier locally, while retaining retrieval in the web app. That remote-backend adapter is not implemented yet, so the current Render web app continues to use the bundled local model.

### Understand responsibility and cleanup

A Model links an image, artifact and execution role. An EndpointConfig defines variants and capacity. An Endpoint is the running hosted resource. A model registration record alone is not a running endpoint. Deleting the endpoint does not delete the source S3 object or ECR image.

```bash
python infra/aws/sagemaker.py delete --region ap-southeast-1 --name bank-lab-v1
```

The helper deletes the explicitly named endpoint, waits, then deletes config/model. If creation failed partway or cleanup is partially complete, it fails visibly; remove the remaining named resources individually in the console. Confirm there are no other exercise endpoints/notebooks/jobs still running. Training, deployment, autoscaling, monitoring and cleanup must all be verified on the real account before claiming AWS experience.

## 38. Cloud CI/CD, monitoring, recovery and cleanup

### How a code change becomes a deployment

The implemented GitHub workflow installs dependencies, runs tests/retrieval checks, rehearses training with synthetic software-check data, builds the image containing the **committed real-data release**, and tests that image over HTTP. `MODEL_PATH=release/model.joblib` ensures the smoke comparator uses the same release. The synthetic model is never shipped in the Docker image. Real-data training remains an optional manual job and local workflow.

Render watches the Git branch and, when configured for CI-pass deployment, builds the checked commit. A normal update does not retrain or silently change the model. A model update is an explicit promotion commit. Infrastructure files and model manifests can therefore be reviewed together, while their release identities remain separate.

A future AWS workflow should use GitHub OIDC to assume a scoped deployment role, not permanent access keys in GitHub. The trust policy restricts the repository and branch or protected environment, the audience should match the AWS token service, and the job needs `id-token: write` plus only necessary repository permissions. Then it publishes an immutable image, deploys staging, checks health/prediction, requests the configured environment approval and promotes production. Keep a previous image digest for rollback. This AWS delivery workflow is designed here but not enabled; no AWS credentials are configured in CI.

### What to monitor on each platform

| Concern | Render demo | AWS deployment |
| --- | --- | --- |
| Startup/process | Deploy logs and health checks | systemd/Docker logs on EC2; endpoint status on SageMaker |
| Availability | `/health`, actual prediction smoke check | External health probe, ALB targets if used, endpoint failures |
| Traffic/performance | Hosting metrics plus application measurements | CloudWatch request/error/latency/CPU/memory metrics as configured |
| Input quality | Future schema/missingness reports | Versioned captures to restricted storage and processing jobs |
| Model performance | Future matured outcome joins | Same logic, scheduled and published as metrics |
| Retrieval | Evidence/no-evidence modes, relevance evaluation | Same logic and knowledge-version tracking |
| Costs | Workspace free usage and provider usage | Billing, resource inventory, budgets and cost allocation tags |

The current `/health` means the service loaded its model; it does not measure business quality. The basic EC2 template does not install a CloudWatch agent or create alarms; those require explicit additional configuration. Similarly, data-drift monitoring is not implemented merely because CloudWatch exists. Follow chapters 25–26 for the data/label logic, then implement and test it before scheduling cloud jobs.

For every alarm, define the observation window, minimum sample size, threshold, owner and response. Examples: sustained prediction errors → inspect release/config and rollback; rising invalid data → fix producer; changed feature distribution → investigate; lower precision after labels mature → compare candidate models. Do not automatically retrain on corrupt inputs or replace a model solely because one drift statistic changed.

### Reliability and scaling progression

Start with one worker and measure memory/latency. A small prototype does not require a GPU or Kubernetes. The local demo is checked with a 512 MB container limit; this is not a capacity guarantee under arbitrary concurrent traffic. Free hosting sleeps and has shared allowances; it is appropriate for a portfolio demo, not an availability commitment.

On EC2, replicas behind an ALB and an Auto Scaling group can improve availability but require compatible releases, stateless serving, traffic checks and additional cost. On SageMaker, endpoint variants and supported scaling mechanisms manage inference capacity, but you must set appropriate policies and quotas. Use staged load tests, examine p95 latency, errors and memory, and account for cold starts/scale-out delays. Preserve batch processing as an alternative when immediate response is unnecessary.

### Teardown is part of the lab

For the EC2 exercise:

```bash
aws cloudformation delete-stack --region ap-southeast-1 --stack-name bank-subscription-lab
aws cloudformation wait stack-delete-complete --region ap-southeast-1 --stack-name bank-subscription-lab
```

This template's root EBS disk is marked DeleteOnTermination. Still inspect for retained snapshots, other disks, ECR images, S3 versions, log groups, IPs, load balancers or resources created separately. ECR/S3 are deliberately outside this EC2 stack. Stop/terminate are different operations; stopped instances can leave billable storage. Do not delete a shared repository or bucket just because a tutorial says “cleanup.” Delete only exercise resources you own, and inspect versioned-bucket contents before removal.

For SageMaker, delete the endpoint/config/model as in chapter 37 and review other jobs/notebooks and artifacts. Budget notifications can be delayed and are not a hard cap. Resource inventory plus actual deletion is the cost control mechanism.

For Render, a rollback should select a known working deploy/commit. To stop hosting, suspend/delete only this project service, leaving unrelated portfolio services alone. Keep source and release evidence in Git. If the free plan requires billing information or the workspace has exhausted free allowances, stop rather than upgrade automatically.

### Interview and practice checkpoints

You should be able to explain why ECR is not S3; why a security group is not an IAM role; why a green CI run is not a healthy deployment; why the app and model have different release histories; why a SageMaker endpoint is not a website; why drift is not necessarily degraded performance; why labelled outcomes may arrive late; and why stopping a VM does not necessarily stop all costs.

Practice a missing model, a bad checksum, a blocked port, an ECR pull denial, a no-evidence query, a failed candidate and an endpoint cleanup. Record the actual error, layer, diagnosis, fix and verification in this single README. Do not claim a resource was deployed, scaled or monitored merely because its template exists.
