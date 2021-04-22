from collections import OrderedDict, defaultdict, namedtuple

import numpy as np
import pytest

from sac.registry import make
from sac.datasets import make_perfect_forecast


def test_make_random_dataset_one_battery():
    env = make('battery', dataset_cfg={'name': 'random-dataset', 'n': 10000, 'n_features': 3})

    dataset = env.dataset.dataset

    assert dataset['prices'].shape[0] == 10000
    assert dataset['features'].shape[0] == 10000

    assert len(dataset['prices'].shape) == 1
    assert dataset['features'].shape[1] == 3


def test_make_random_dataset_many_battery():
    env = make(
        'battery',
        n_batteries=4,
        dataset_cfg={
            'name': 'random-dataset',
            'n': 1000,
            'n_features': 6,
            'n_batteries': 4
        }
    )

    data = env.dataset.dataset
    print(data['prices'].shape, data['features'].shape)
    assert data['prices'].shape[0] == 1000

    #  (timestep, feature, battery)
    assert data['features'].shape[0] == 1000
    assert data['features'].shape[1] == 6
    assert data['features'].shape[2] == 4


def test_make_perfect_forecast():
    prices = np.array([10, 50, 90, 70, 40])
    horizon = 3
    forecast = make_perfect_forecast(prices, horizon)

    expected = np.array([
        [10, 50, 90],
        [50, 90, 70],
        [90, 70, 40],
    ])
    np.testing.assert_array_equal(expected, forecast)


def test_make_nem_dataset_one_battery():
    #  TODO sync s3 with my clean data

    dataset = make('nem-dataset')

    #  check we get same / different data
    stats = defaultdict(list)
    for ep in range(16):
        dataset.reset()
        stats['avg-price'].append(np.mean(dataset.dataset['prices']))

    assert all(np.mean(stats['avg-price']) != dataset.dataset['prices'])

    #  check correct length
