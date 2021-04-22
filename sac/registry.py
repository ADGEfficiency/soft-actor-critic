from sac.envs.battery import Battery
from sac.envs.gym_wrappers import GymWrapper

from sac.datasets import *


registry = {
    'lunar': GymWrapper,
    'pendulum': GymWrapper,

    'battery': Battery,

    'random-dataset': RandomDataset,
    'nem-dataset': NEMDataset,
}


def make(name=None, *args, **kwargs):
    if name is None:
        name = kwargs['name']
    return registry[name](*args, **kwargs)
