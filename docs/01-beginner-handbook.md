# Bank Customer Subscription ML Deployment — Beginner Handbook

## 1. The problem and our first milestone

A bank has limited time to call customers about a term deposit. We want a score that helps rank potential customers before a call. Historical rows tell us what happened in earlier campaigns. A model learns associations in those rows.

The first milestone is deliberately small: get one customer prediction on your laptop. Later milestones expose the same prediction over HTTP, package it, and automate checks. AWS comes after these work locally.

A prediction is an estimate, not a promise or proof of what caused a customer to subscribe. This historical dataset is not evidence that the model is ready for today's banking customers.

## 2. Your tools

The terminal is an application where you type commands. Its current directory determines where files are read and written. Open a terminal in this project's root folder, the folder containing `pyproject.toml`.

On macOS, `pwd` prints the current folder and `ls` lists its contents. A command printed in a code block is something you can run; a filename in a paragraph is something you can open and read.

You need Python 3.10 or newer. Docker is needed only for the container lesson. Git records changes; GitHub hosts the repository online. You do not need an AWS account for these lessons.

```bash
python3 --version
pwd
ls
```

Expected: a supported Python version and project files. If a command is missing, use the troubleshooting guide rather than installing random packages.

## 3. Create an isolated Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

`python3 -m venv .venv` creates a private set of Python tools in `.venv`. It does not create a virtual machine. `source` activates those tools in this terminal; activate again when you open another terminal. `python -m` runs a module using the selected Python interpreter.

`pip` installs libraries. `-e` means editable: when you change our Python source, the environment sees the changes. The quoted `.[dev]` means this project plus its development tools. Quotes prevent the shell treating brackets as filename patterns. `pyproject.toml` declares those requirements. Version ranges allow compatible updates; this is not a fully locked dependency environment.

Expected: installation finishes without an error. The `.venv` directory is ignored by Git because it is generated and specific to your machine. Do not use `sudo pip`.

## 4. Obtain the data

```bash
python -m bank_ml.data
```

The program downloads a public ZIP from UCI, reads the original `bank-full.csv` from its nested ZIP, and saves it in `data/`. It also writes a `.source.json` file with a source URL, license and SHA-256 checksum. The checksum is a fingerprint: different bytes produce a different fingerprint. It records what we downloaded; it is not an independently verified authenticity signature.

Expected: `data/bank-full.csv` and `data/bank-full.source.json`. The CSV uses semicolons, so our reader explicitly uses `sep=";"`. This is real historical learning data, not a random fixture. Tests use generated examples so ordinary tests do not need internet access.

Read the data card before changing the feature list. Exercise: open the CSV and locate `y`, the answer column, and `duration`, a column we deliberately exclude.

## 5. Train and evaluate

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

## 6. Predict one customer

```bash
python -m bank_ml.predict --input examples/customer.json
```

JSON is a text format made of named fields and values. Our example customer is fictional. The program loads the saved model, checks the inputs, and prints a probability between zero and one. It does not retrain.

The probability is uncalibrated and based on this historical training distribution. Do not interpret a score as guaranteed real-world odds.

Exercise: copy the example into another local JSON file and change a field. Run the prediction with `--input` pointing at that file. A changed result is an association learned by the model, not proof that changing that characteristic causes subscription.

## 7. Serve the prediction through an API

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

## 8. Run automated checks

```bash
python -m pytest -q
```

Tests are programs that check expected behaviour. We test the full training/save/load/predict path, valid and invalid API requests, unfamiliar job categories, exclusion of call duration, and failure when the model is missing. These tests check software behaviour, not whether the model is commercially useful or fair.

Exercise: send an age of -1 to the API. Explain why it is better to reject it than silently produce a score.

## 9. Package and run with Docker

First train the model. Start Docker Desktop and wait until its engine is ready.

```bash
docker build -t bank-ml:local .
docker run --rm --name bank-ml -p 127.0.0.1:8000:8000 bank-ml:local
```

The build reads `Dockerfile`, installs the application, and includes the trained model. The image is a packaged template; the container is a running instance. `.dockerignore` limits which local files enter the build context. The application runs as a non-root user.

The port mapping forwards laptop port 8000 to container port 8000. The server listens on `0.0.0.0` inside the container so Docker can reach it; the published laptop port is restricted to localhost. Stop any Python API already using that port first.

Run the same curl requests. Stop the container with Control+C. `--rm` removes this stopped container; the reusable image remains. Retraining a local model does not alter an existing image: rebuild it to include the new model.

A container provides a consistent application environment. It does not automatically provide IAM permissions, TLS, monitoring or production security.

## 10. Understand Git and GitHub

Git is local history. GitHub stores a remote copy and runs automation. A commit is a named snapshot of tracked files. Generated data, environments, models and credentials should not be in those snapshots.

```bash
git status
git diff
git add src docs tests README.md
git commit -m "Explain the change you made"
git push
```

Read `git diff` before staging. `git add` selects changes; `git commit` saves them locally; `git push` sends commits to the configured remote. The first publication is handled as part of project setup. Do not rerun repository creation commands for every change.

## 11. Understand our CI workflow

Open `.github/workflows/ci.yml`. A workflow is a YAML file describing automation. A job runs on a temporary GitHub-hosted Linux machine. Each step executes a command or reusable action.

On a push, pull request, or manual dispatch, it checks out the code, installs Python, runs tests, downloads data, trains the model, predicts an example, builds Docker and checks the running container. It stops when a required command fails. A cleanup trap removes the temporary container.

View progress under the repository's Actions tab. A green run means those checks passed for that commit. It does not mean AWS deployment succeeded: this workflow never deploys to AWS. Downloads can fail if UCI is unavailable; read the failed step's log.

The repository is public. Standard GitHub-hosted runners are free for public repositories; other runner types and storage have separate billing rules. The workflow has a 15-minute timeout and read-only repository permissions. It does not upload large datasets or models as artifacts.

## 12. How we continue

Complete the lessons in order. At each stage, explain the input, action and output in your own words. Then make one controlled change and inspect the effect.

Next we will add S3 API practice, model versioning, a local release and rollback exercise, monitoring and retraining. Only then will we prepare and optionally execute real AWS deployment. The AWS roadmap explains the differences between a local substitute and a verified cloud service.
