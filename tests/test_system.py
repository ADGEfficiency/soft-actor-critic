from sac.main import main


hyp = {
    'initial-log-alpha': 0.0,
    'gamma': 0.99,
    'rho': 0.995,
    'buffer-size': int(1e3),
    'reward-scale': 5,
    'lr': 3e-4,
    'batch-size': 1024,
    'n-episodes': 1,
    'test-every': 1,
    'n-tests': 3,
    'size-scale': 1,
    'env-name': 'pendulum',
    'run-name': 'test',
    'buffer': 'new'
}


def test_system():
    main(hyp)


if __name__ == '__main__':
    test_system()
