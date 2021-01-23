from sac.env import GymWrapper
from sac.random_policy import make as make_random_policy


def test_envs():
    env = GymWrapper('pendulum')
    policy = make_random_policy(env)

    env = GymWrapper('lunar')
    policy = make_random_policy(env)
