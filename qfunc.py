
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from env import env

def make_qfunc(obs_shape, n_actions):

    in_obs = keras.Input(shape=obs_shape)
    in_act = keras.Input(shape=n_actions)
    inputs = tf.concat([in_obs, in_act], axis=1)

    net = layers.Dense(64, activation='relu')(inputs)
    q_value = layers.Dense(1, activation='linear')(net)

    return keras.Model(
        inputs=[in_obs, in_act],
        outputs=q_value
    )


def update_params(online, target, rho):
    for o, t in zip(online.trainable_variables, target.trainable_variables):
        t.assign(rho * t.value() + (1 - rho) * o.value())


def test_update_params():
    from numpy.testing import assert_array_equal, assert_raises
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
    update_params(online, target, 0.0)
    for o, t in zip(online.trainable_variables, target.trainable_variables):
        assert_array_equal(o.value(), t.value())


def setup_dummy_qfunc():
    import numpy as np
    obs = np.random.uniform(0, 1, 6).reshape(2, 3)
    act = np.random.uniform(0, 1, 4).reshape(2, 2)
    return make_qfunc((3, ), (2, )), obs, act


if __name__ == '__main__':
    print('model outputs:\n')
    model, obs, act = setup_dummy_qfunc()
    print(model([obs, act]))
    test_update_params()
