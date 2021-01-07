from collections import namedtuple

import gym

env_ids = [
    'Pendulum-v0'
    'LunarLanderContinuous-v2',
]


def inverse_scale(action, low, high):
    return action * (high - low) + low


class GymWrapper():
    def __init__(self, env_id=:
        self.env = gym.make(env_id)
        self.elements = (
            ('observation', self.env.observation_space.shape, 'float32'),
            ('action', self.env.action_space.shape, 'float32'),
            ('reward', (1, ), 'float32'),
            ('next_observation', self.env.observation_space.shape, 'float32'),
            ('done', (1, ), 'bool')
        )
        self.Transition = namedtuple('Transition', [el[0] for el in self.elements])
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        #  expect a scaled action here
        assert action <= 1
        assert action >= -1
        unscaled_action = action * self.env.action_space.high
        next_obs, reward, done, _ = self.env.step(unscaled_action)
        return next_obs.reshape(1, *self.env.observation_space.shape), reward, done

    def reset(self):
        return self.env.reset().reshape(1, *self.env.observation_space.shape)


if __name__ == '__main__':
    lunar = GymWrapper()
