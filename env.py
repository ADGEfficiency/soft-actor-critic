from collections import namedtuple

import gym

env = gym.make('Pendulum-v0')

elements = (
    ('observation', env.observation_space.shape, 'float32'),
    ('action', env.action_space.shape, 'float32'),
    ('reward', (1, ), 'float32'),
    ('next_observation', env.observation_space.shape, 'float32'),
    ('done', (1, ), 'bool')
)

Transition = namedtuple('Transition', [el[0] for el in elements])
