# Common problems and fixes

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
| Docker build cannot find model.joblib | Model not trained before build | Train first; inspect artifacts/model.joblib |
| Git push says authentication failed | GitHub login missing or invalid | Run `gh auth login` interactively; do not put credentials in repository files |
| CI fails in download step | External dataset source unavailable | Read the logs and rerun later; tests themselves are offline |
| CI does not start | Actions disabled or billing allowance exhausted | Check repository Actions settings and account billing |

## Collect useful debugging information

Run from the project root:

```bash
pwd
python --version
python -m pip --version
python -m pytest -q
git status
```

For containers, run `docker ps` to see running containers and `docker logs bank-ml` to read logs while the named container exists. Do not solve a port problem by opening your machine to the internet.

## Rebuild safely

A virtual environment can be recreated from project dependencies. Do not delete project source, datasets or Git history while troubleshooting. Generated models can be recreated by training. A model trained in one library environment may not load in a different one: rebuild it using the intended runtime.
