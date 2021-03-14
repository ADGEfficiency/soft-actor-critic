from collections import OrderedDict, defaultdict

import numpy as np
import pytest

from sac.registry import make


test_cases = (
    #  full charge for three steps
    (
        {'initial_charge': 0.0, 'power': 2.0, 'capacity': 100, 'episode_length': 3},
        [1.0, 1.0, 1.0],
        [2.0/12, 4.0/12, 6.0/12]
    ),
    #  full, half then full charge for three steps
    (
        {'initial_charge': 0.0, 'power': 2.0, 'capacity': 100, 'episode_length': 3},
        [1.0, 0.5, 1.0],
        [2.0/12, 3.0/12, 5.0/12]
    ),
    (
    #   discharge, charge, discharge
        {'initial_charge': 0.0, 'power': 2.0, 'capacity': 100, 'episode_length': 3},
        [-1.0, 1.0, -1.0],
        [0.0, 2.0/12, 0.0]
    )
)


@pytest.mark.parametrize('cfg, actions, charges', test_cases)
def test_one_battery_charging(cfg, actions, charges):
    env = make('battery', **cfg)
    env.reset()

    results = defaultdict(list)
    for action in actions:
        next_obs, reward, done, info = env.step(action)
        results['charge'].append(next_obs['charge'])

    assert done
    np.testing.assert_array_almost_equal(results['charge'], charges)


def test_make_random_dataset_one_battery():
    env = make('battery', dataset_cfg={'name': 'random-dataset', 'n': 10000, 'n_features': 3})

    assert env.dataset['prices'].shape[0] == 10000
    assert env.dataset['features'].shape[0] == 10000

    assert env.dataset['prices'].shape[1] == 1
    assert env.dataset['features'].shape[1] == 3


def test_make_random_dataset_many_battery():
    env = make(
        'many-battery',
        n_batteries=4,
        dataset_cfg={'name': 'random-dataset', 'n': 1000, 'n_features': 6}
    )

    data = env.dataset
    print(data['prices'].shape, data['features'].shape)
    assert data['prices'].shape[0] == 1000

    assert data['features'].shape[0] == 1000
    assert data['features'].shape[1] == 6


def test_many_battery_step():

    cfgs = defaultdict(list)

    actions, charges = [], []
    for test_case in test_cases:

        #  the config dict
        for k, v in test_case[0].items():
            cfgs[k].append(v)

        actions.append(test_case[1])
        charges.append(test_case[2])

    #  actions = (3, 3)
    #  needs to be timestep first!
    actions = np.array(actions).T
    expected_charges = np.array(charges).T

    env = make('many-battery', n_batteries=len(test_cases), **cfgs)

    #  test 1
    np.testing.assert_array_equal(cfgs['power'], env.power[0, :, 0])
    assert env.power.shape == (1, len(test_cases), 1)

    obs = env.reset()
    results = defaultdict(list)
    for action in actions:
        action = np.array(action).reshape(1, len(test_cases), 1)
        next_obs, reward, done, info = env.step(action)
        print(env.charge, 'charge')
        results['charge'].append(next_obs['charge'])

    assert all(done)
    np.testing.assert_array_almost_equal(
        np.squeeze(results['charge']),
        np.squeeze(expected_charges)
    )
