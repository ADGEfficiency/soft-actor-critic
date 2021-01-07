from env import GymWrapper, env_ids
from random_policy import make_random_policy


if __name__ == '__main__':
    env = GymWrapper(env_ids['pendulum'])
    policy = make_random_policy(env)

    env = GymWrapper(env_ids['lunar'])
    policy = make_random_policy(env)
