from collections import namedtuple

import gym


#  key=name, value=id
env_ids = {
    'pendulum': 'Pendulum-v0',
    'lunar': 'LunarLanderContinuous-v2'
}


def inverse_scale(action, low, high):
    return action * (high - low) + low


class GymWrapper():
    def __init__(self, env_name, monitor=False):
        self.env_id = env_ids[env_name]
        self.env = gym.make(self.env_id)
        if monitor:
            self.env = gym.wrappers.Monitor(self.env, monitor, force=True)
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
        assert action.all() <= 1
        assert action.all() >= -1
        unscaled_action = action * self.env.action_space.high
        if 'lunar' in self.env_id.lower():
            unscaled_action = unscaled_action.reshape(-1)
        next_obs, reward, done, _ = self.env.step(unscaled_action)
        return next_obs.reshape(1, *self.env.observation_space.shape), reward, done

    def reset(self):
        return self.env.reset().reshape(1, *self.env.observation_space.shape)
