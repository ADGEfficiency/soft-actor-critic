from collections import namedtuple

import gym




class GymWrapper():
    def __init__(self, env_id='Pendulum-v0'):
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
        next_obs, reward, done, _ = self.env.step(action)
        return next_obs.reshape(1, *self.env.observation_space.shape), reward, done

    def reset(self):
        return self.env.reset().reshape(1, *self.env.observation_space.shape)
