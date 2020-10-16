import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import minimum_target


def make_qfunc(obs_shape, n_actions, name):

    in_obs = keras.Input(shape=obs_shape)
    in_act = keras.Input(shape=n_actions)
    inputs = tf.concat([in_obs, in_act], axis=1)

    net = layers.Dense(64, activation='relu')(inputs)
    q_value = layers.Dense(1, activation='linear')(net)

    return keras.Model(
        inputs=[in_obs, in_act],
        outputs=q_value,
        name=name
    )


def update_target_network(online, target, rho, step=None, writer=None):
    for o, t in zip(online.trainable_variables, target.trainable_variables):
        t.assign(rho * t.value() + (1 - rho) * o.value())

    if writer:
        with writer.as_default():
            tf.summary.histogram(f'{target.name} target weights', target.trainable_variables[-2], step=step)


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
    update_target_network(online, target, 0.0)
    for o, t in zip(online.trainable_variables, target.trainable_variables):
        assert_array_equal(o.value(), t.value())


def setup_dummy_qfunc():
    import numpy as np
    obs = np.random.uniform(0, 1, 6).reshape(2, 3)
    act = np.random.uniform(0, 1, 4).reshape(2, 2)
    return make_qfunc((3, ), (2, )), obs, act


def initialize_qfuncs(env):
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.shape

    q1 = make_qfunc(obs_shape, n_actions, 'q1')
    q1_target = make_qfunc(obs_shape, n_actions, 'q1-target')
    q2 = make_qfunc(obs_shape, n_actions, 'q2')
    q2_target = make_qfunc(obs_shape, n_actions, 'q2-target')

    update_target_network(online=q1, target=q1_target, rho=0.0)
    update_target_network(online=q2, target=q2_target, rho=0.0)
    onlines = [q1, q2]
    targets = [q1_target, q2_target]
    return onlines, targets


def update_qfuncs(batch, actor, onlines, targets, writer, optimizer, counters, hyp):
    next_state_act, log_prob, _ = actor(batch['next_observation'])
    next_state_target = minimum_target(batch['next_observation'], next_state_act, targets)

    al = hyp['alpha']
    ga = hyp['gamma']
    target = batch['reward'] + ga * (1 - batch['done']) * (next_state_target - al * log_prob)

    for onl in onlines:
        with tf.GradientTape() as tape:
            loss = tf.keras.losses.MSE(
                onl([batch['observation'], batch['action']]), target
            )

        grads = tape.gradient(loss, onl.trainable_variables)
        optimizer.apply_gradients(zip(grads, onl.trainable_variables))

        with writer.as_default():
            step = counters['qfunc_updates']
            tf.summary.scalar('qfunc target', tf.reduce_mean(target), step=step)
            tf.summary.scalar(f'{onl.name} loss', tf.reduce_mean(loss), step=step)
            tf.summary.histogram(f'{onl.name} weights', onl.trainable_variables[-2], step=step)
            counters['qfunc_updates'] += 1


if __name__ == '__main__':
    print('model outputs:\n')
    model, obs, act = setup_dummy_qfunc()
    print(model([obs, act]))
    test_update_params()
