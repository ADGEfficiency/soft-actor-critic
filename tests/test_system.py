from sac.main import main
from sac.init import init_fresh

from pathlib import Path
from shutil import rmtree


hyp = {
    'initial-log-alpha': 0.0,
    'gamma': 0.99,
    'rho': 0.995,
    'buffer-size': int(1e3),
    'reward-scale': 5,
    'lr': 3e-4,
    'batch-size': 4,
    'n-episodes': 2,
    'test-every': 1,
    'n-tests': 1,
    'size-scale': 1,
    'env-name': 'pendulum',
    'run-name': 'test-system',
    'delete-previous': True,
    'buffer': 'new'
}


def test_system():
    main(**init_fresh(hyp))


if __name__ == '__main__':
    test_system()
