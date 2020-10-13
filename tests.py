from collections import namedtuple

import numpy as np

from buffer import Buffer


def test_buffer():
    elements = (
        ('A', (1,), 'int32'),
        ('B', (2,), 'float32'),
    )

    tup = namedtuple('tup', ('A', 'B'))

    buff = Buffer(elements, size=4)

    data = ((0, (1, 1)), (1, (2, 2)), (2, (3, 3)))
    data = [tup(*d) for d in data]

    for _ in range(2):
        for d in data:
            buff.append(d)

    batch = buff.sample(2)

    for a, b in zip(batch['A'], batch['B']):
        check = data[int(a)]
        np.testing.assert_array_equal(b, check[1])

    np.testing.assert_array_equal(buff.data['A'], np.array([1, 2, 2, 0]).reshape(4, 1))
    np.testing.assert_array_equal(buff.data['B'], np.array(((2, 2), (3, 3), (3, 3), (1, 1))).reshape(4, 2))



def test_system():
    #from env import env, elements
    from buffer import Buffer
    from policy import make_random_policy, make_policy
    from main import episode

    env = GymWrapper('Pendulum-v0')
    buffer = Buffer(env.elements, size=1024)
    random_policy = make_random_policy(env)
    while not buffer.full:
        buffer = episode(env, buffer, random_policy)
    batch = buffer.sample(64)

    policy = make_policy(env)
    for _ in range(3):
        buffer = episode(env, buffer, policy)


from env import GymWrapper
def test_pendulum_wrapper():
    env = GymWrapper('Pendulum-v0')

    res = env.reset()
    assert res.shape == (1, 3)

    act = env.action_space.sample().reshape(1, 1)

    next_obs, rew, done = env.step(act)
    assert next_obs.shape == (1, 3)


from policy import make_random_policy
def test_random_policy_wrapper():
    env = GymWrapper('Pendulum-v0')
    pol = make_random_policy(env)
    obs = env.reset()
    sample, _ , _ = pol(obs)
    assert sample.shape == (1, 1)
