from collections import OrderedDict, defaultdict, namedtuple

from pathlib import Path
import numpy as np
import pandas as pd


class RandomDataset:
    def __init__(self, n=1000, n_features=3, n_batteries=1):
        self.dataset = self.make_random_dataset(n, n_features, n_batteries)

    def make_random_dataset(self, n, n_features, n_batteries):
        np.random.seed(42)
        #  (timestep, features, batteries)
        prices = np.random.uniform(0, 100, n).reshape(-1, )
        features = np.random.uniform(0, 100, n*n_features*n_batteries).reshape(n, n_features, n_batteries)
        return {
            'prices': prices,
            'features': features
        }

    def get_data(self, cursor):
        return OrderedDict(
            {k: d[cursor] for k, d in self.dataset.items()}
        )

    def reset(self):
        pass


def make_perfect_forecast(prices, horizon):
    prices = np.array(prices).reshape(-1, 1)
    forecast = np.hstack([np.roll(prices, -i) for i in range(0, horizon)])
    #  +1 because we include the current price in the forecast
    return forecast[:-(horizon-1), :]


class NEMDataset:
    def __init__(
        self,
        n_batteries=1,
        episode_length=128
    ):
        assert n_batteries == 1, 'dont support more than one'
        self.episode_length = episode_length

        self.datasets = {
            'train': self.make_nem_dataset_one_battery(),
            'test': self.make_nem_dataset_one_battery()
        }

        self.reset()

    def make_nem_dataset_one_battery(self):
        horizon = 12

        prices = [p / 'clean.csv' for p in (Path.home() / 'nem-data' / 'trading-price').iterdir()]

        #  NOT SELECTING REGION

        prices = [pd.read_csv(p, index_col='interval-start', parse_dates=True) for p in prices]
        prices = pd.concat([p[['trading-price', 'REGIONID']] for p in prices], axis=0)

        region = 'SA1'
        region_mask = prices['REGIONID'] == region
        prices = prices.loc[region_mask, :]
        assert len(set(prices['REGIONID'])) == 1
        prices = prices.drop('REGIONID', axis=1)

        datetime = prices.index.values[:-(horizon-1)]
        features = make_perfect_forecast(prices['trading-price'], horizon)
        prices = prices.values[:-(horizon-1), :]

        assert prices.shape[0] == features.shape[0]
        assert prices.shape[0] == datetime.shape[0]

        return {
            'prices': prices,
            'features': features,
            'datetime': datetime
        }

    def get_data(self, cursor):
        return OrderedDict(
            {k: d[cursor] for k, d in self.dataset.items()}
        )

    def reset(self, mode='train'):
        self.dataset = self.sample_episode(mode)

    def sample_episode(self, mode):
        #  mode could also be dataset
        dataset = self.datasets[mode]
        pop_len = dataset['prices'].shape[0]

        start = np.random.randint(0, pop_len - self.episode_length, 1)[0]
        end = start + self.episode_length

        episode = {}
        for name, data in dataset.items():
            episode[name] = data[start:end]

        print(f' sampled nem episode')
        return episode
