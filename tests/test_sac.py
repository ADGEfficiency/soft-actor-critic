from collections import namedtuple

import numpy as np
from numpy.testing import assert_array_equal

from sac.memory import Buffer
from sac.envs.gym_wrappers import GymWrapper
from sac.random_policy import make as make_random_policy
from sac.qfunc import make_qfunc
from sac.qfunc import update_target_network


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



def test_pendulum_wrapper():
    env = GymWrapper('pendulum')
    res = env.reset()
    assert res.shape == (1, 3)
    act = env.action_space.sample().reshape(1, 1)
    next_obs, rew, done, _ = env.step(act)
    assert next_obs.shape == (1, 3)


def test_random_policy_wrapper():
    env = GymWrapper('pendulum')
    pol = make_random_policy(env)
    obs = env.reset()
    sample, _ , _ = pol(obs)
    assert sample.shape == (1, 1)


def setup_dummy_qfunc():
    import numpy as np
    obs = np.random.uniform(0, 1, 6).reshape(2, 3)
    act = np.random.uniform(0, 1, 4).reshape(2, 2)
    return make_qfunc((3, ), (2, ), 'dummy'), obs, act


def test_update_params():
    online, _, _ = setup_dummy_qfunc()
    target, _, _ = setup_dummy_qfunc()

    #  check to see that some params are different
    #  can't do for all as biases are init to zero
    diff_check = False
    for o, t in zip(online.trainable_variables, target.trainable_variables):
        same = o.value().numpy() == t.value().numpy()
        if not same.any():
            diff_check = True
    assert diff_check

    #  check to see they are all the same
    update_target_network(online, target, 0.0)
    for o, t in zip(online.trainable_variables, target.trainable_variables):
        assert_array_equal(o.value(), t.value())
