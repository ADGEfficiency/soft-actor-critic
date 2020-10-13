import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_probability as tfp


#  clip as per stable baselines
log_stdev_low, log_stdev_high = -2, 20
epsilon = 1e-6


class RandomPolicy():
    def __init__(self, env):
        self.env = env
        assert abs(self.env.action_space.low) == abs(self.env.action_space.high)
    def __call__(self, observation=None):
        unscaled = self.env.action_space.sample().reshape(1, *self.env.action_space.shape)
        scaled = unscaled / abs(self.env.action_space.high)
        return scaled, None, None

def make_random_policy(env):
    return RandomPolicy(env)


def make_policy(env):
    obs = env.reset().reshape(1, -1)
    obs_shape = obs.shape[1:]
    n_actions = env.action_space.shape[0]

    inputs = keras.Input(shape=obs_shape)
    net = layers.Dense(64, activation='relu')(inputs)
    net = layers.Dense(n_actions*2, activation='linear')(net)

    mean, log_stdev = tf.split(net, 2, axis=1)

    log_stdev = tf.clip_by_value(log_stdev, log_stdev_low, log_stdev_high)

    stdev = tf.exp(log_stdev)

    normal = tfp.distributions.Normal(
        mean, stdev, allow_nan_stats=False
    )

    #  unsquashed
    action = normal.sample()
    log_prob = normal.log_prob(action)

    #  squashed
    action = tf.tanh(action)
    deterministic_action = tf.tanh(mean)
    log_prob -= tf.reduce_sum(
        tf.math.log(1 - action ** 2 + epsilon),
        axis=1,
        keepdims=True
    )

    model = keras.Model(
        inputs=inputs,
        outputs=[action, log_prob, deterministic_action]
    )
    return model
