setup:
	pip install -r requirements.txt
	pip install .

test:
	pytest tests --tb=line --disable-pytest-warnings

tensorboard:
	tensorboard --logdir experiments/results
