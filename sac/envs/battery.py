from collections import OrderedDict, defaultdict, namedtuple

import numpy as np

from sac import registry


def battery_energy_balance(
    initial_charge, final_charge, import_energy, export_energy, losses
):
    delta_charge = final_charge - initial_charge
    balance = import_energy - (export_energy + losses + delta_charge)
    np.testing.assert_almost_equal(balance, 0)


def calculate_losses(delta_charge, efficiency):
    #  delta_charge = battery delta_charge charge
    delta_charge = np.array(delta_charge)
    efficiency = np.array(efficiency)
    #  account for losses / the round trip efficiency
    #  we lose electricity when we discharge
    losses = delta_charge * (1 - efficiency)
    losses = np.array(losses)
    losses[delta_charge > 0] = 0
    return np.abs(losses)


class BatteryObservationSpace:
    def __init__(self, dataset):
        self.shape = dataset.dataset['features'].shape[1:]


class BatteryActionSpace:
    def __init__(self, dataset):
        self.shape = dataset.dataset['prices'].shape[1:]
        self.low = np.array((-1, ))
        self.high = np.array((-1, ))

    #  needs the same dataset
    def sample(self):
        return np.random.uniform(0, 1, 1)


class Battery:
    def __init__(
        self,
        power=2.0,
        capacity=4.0,
        efficiency=0.9,
        initial_charge=0.0,
        episode_length=288,
        dataset_cfg={'name': 'random-dataset', 'n_features': 4}
    ):
        self.power = float(power)
        self.capacity = float(capacity)
        self.efficiency = float(efficiency)
        self.initial_charge = float(initial_charge)
        self.episode_length = int(episode_length)

        #  TODO
        self.dataset = registry.make(**dataset_cfg)
        self.observation_space = BatteryObservationSpace(self.dataset)
        self.action_space = BatteryActionSpace(self.dataset)

        self.elements = (
            ('observation', self.observation_space.shape, 'float32'),
            ('action', self.action_space.shape, 'float32'),
            ('reward', (1, ), 'float32'),
            ('next_observation', self.observation_space.shape, 'float32'),
            ('done', (1, ), 'bool')
        )
        self.Transition = namedtuple('Transition', [el[0] for el in self.elements])

    def __repr__(self):
        return f'<energypy Battery: {self.power:2.1f} MW {self.capacity:2.1f} MWh>'

    def reset(self):
        len_dataset = 1000

        self.start = np.array(0).astype(int)
        self.cursor = np.copy(self.start)
        self.charge = self.initial_charge

        data = self.dataset.get_data(self.cursor)
        self.cursor += 1
        return data['features']

    def step(self, action):
        """action > 0 to charge, action < 0 to discharge"""
        #  rewrite to be import / export, losses

        #  expect a scaled action here
        #  -1 = discharge max, 1 = charge max
        action = np.clip(action, -1, 1)
        action = action * self.power

        #  convert from power to energy, kW -> kWh
        action = action / 12

        #  charge at the start of the 5 min interval, kWh
        initial_charge = self.charge

        #  charge at end of the interval
        #  clipped at battery capacity, kWh
        final_charge = np.clip(initial_charge + action, 0, self.capacity)

        #  accumulation in battery, kWh
        #  delta_charge can also be thought of as gross_power
        delta_charge = final_charge - initial_charge

        #  losses are applied when we discharge, kWh
        losses = calculate_losses(delta_charge, self.efficiency)

        #  net of losses, kWh
        #  add losses here because in delta_charge, export is negative
        #  to reduce export, we add a positive losses
        net_energy = delta_charge + losses

        import_energy = np.zeros_like(net_energy)
        import_energy[net_energy > 0] = net_energy[net_energy > 0]

        export_energy = np.zeros_like(net_energy)
        export_energy[net_energy < 0] = np.abs(net_energy[net_energy < 0])

        #  set charge for next timestep
        self.charge = initial_charge + delta_charge

        battery_energy_balance(
            initial_charge,
            final_charge,
            import_energy,
            export_energy,
            losses
        )

        price = self.dataset.dataset['prices'][self.cursor]
        assert price.shape == (1, )

        reward = export_energy * price - import_energy * price

        self.cursor += 1
        next_obs = self.dataset.get_data(self.cursor)['features'].reshape(1, -1)
        done = int(self.cursor - self.start) == self.episode_length + 1

        info = {
            'start': self.start,
            'cursor': self.cursor,
            'done': done,
            'charge': self.charge
        }

        return next_obs, reward, done, info
