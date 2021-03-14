from sac.envs.battery import *
from sac import registry


class ManyBatteryActionSpace:
    # def sample(self):
    #     return np.random.uniform(0, 1, 1)
    def contains(self, action):
        assert (action <= 1.0).all()
        assert (action >= -1.0).all()
        return True


class ManyBatteries:
    """
    data = (n_battery, timesteps, features)
    """
    def __init__(
        self,
        n_batteries=2,
        power=2.0,
        capacity=4.0,
        efficiency=0.9,
        initial_charge=0.0,
        episode_length=288,
        dataset_cfg={'name': 'random-dataset'}
    ):
        self.n_batteries = n_batteries

        #  we could require everything to be in 3 d
        #  (timestep, battery, features)

        #  currently power is (battery, 1) - what does 1 mean?
        #  could do full like dataset['prices']
        self.power = set_battery_config(power, n_batteries)
        self.capacity = set_battery_config(capacity, n_batteries)
        self.efficiency = set_battery_config(efficiency, n_batteries)
        self.initial_charge = set_battery_config(initial_charge, n_batteries)

        #  TODO
        episode_length = set_battery_config(episode_length, n_batteries)
        self.episode_length = episode_length.tolist()[0][0][0]

        dataset_cfg['n_batteries'] = n_batteries
        self.dataset = registry.make(**dataset_cfg)

        self.action_space = ManyBatteryActionSpace()

    def get_data(self):
        return OrderedDict(
            {k: d[self.cursor] for k, d in self.dataset.items()}
        )

    def reset(self):
        len_dataset = 1000
        self.start = np.random.randint(
            0, len_dataset - self.episode_length, self.n_batteries
        )
        self.cursor = np.copy(self.start)
        self.charge = self.initial_charge
        return None

    def step(self, action):
        assert action.shape == (1, self.n_batteries, 1)

        #  expect a scaled action here
        assert self.action_space.contains(action)

        action = action * self.power
        #  charge at the start of the 5 min interval
        initial_charge = self.charge

        #  convert from power to energy
        action = action / 12
        #  charge at end of the interval
        #  clipped at battery capacity
        final_charge = np.clip(initial_charge + action, 0, self.capacity)

        #  no losses
        gross_power = (final_charge - initial_charge) * 12

        #  set charge for next timestep
        self.charge = initial_charge + (gross_power / 12)
        print(action, gross_power, self.charge)
        print('')

        losses = calculate_losses(gross_power, self.efficiency)

        #  net power is important because this is used for reward
        net_power = gross_power + losses

        price = self.dataset['prices'][self.cursor].reshape(-1, 1, 1)

        #  all batteries, one timestep, one feature (price)
        assert price.shape == (self.n_batteries, 1, 1)
        reward = -1 * net_power * price

        self.cursor += 1

        """
        needs thinking

        - either data changes (cursor position fixed relative) [0, 0, 0] - cursor could be a single number
        - or entire dataset, with cursor [5, 9, 22]

        """
        next_obs = self.get_data()
        next_obs['charge'] = self.charge

        done = (self.cursor - self.start) == self.episode_length

        #  useful for logs
        info = {
            'start': self.start,
            'cursor': self.cursor,
            'done': done
        }

        return next_obs, reward, done, info


