setup-dev-venv:
	python -m venv ./venv && source venv/bin/activate && pip install -r src/requirements.txt && pip install -r src/requirements-dev.txt
