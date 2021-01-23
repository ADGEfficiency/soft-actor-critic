test:
	pytest test_system.py --tb=line --disable-pytest-warnings

tensorboard:
	tensorboard --logdir experiments/results
