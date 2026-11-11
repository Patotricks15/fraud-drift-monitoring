.PHONY: install test notebook reproduce

install:
	pip install -e ".[dev]"

test:
	pytest

notebook:
	jupytext --to ipynb notebooks/fd06_drift_monitoring.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/fd06_drift_monitoring.ipynb

reproduce: test notebook
