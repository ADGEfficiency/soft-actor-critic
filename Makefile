.PHONY: tests

setup:
	pip install -r requirements.txt
	pip install .

tests:
	pytest tests --tb=line --disable-pytest-warnings

tensorboard:
	tensorboard --logdir experiments
