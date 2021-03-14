from sac.envs.battery import *
from sac.envs.parallel_battery import *
from sac.envs.gym_wrappers import GymWrapper


registry = {
    'battery': Battery,
    'random-dataset': make_random_dataset,
    'many-battery': ManyBatteries,

    'lunar': GymWrapper,
    'pendulum': GymWrapper,
}


def make(name=None, *args, **kwargs):
    if name is None:
        name = kwargs['name']
    return registry[name](*args, **kwargs)
