import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

from utils import minimum_target


#  clip as per stable baselines
log_stdev_low, log_stdev_high = -2, 20
epsilon = 1e-6


def make_policy(env):
    obs = env.reset().reshape(1, -1)
    obs_shape = obs.shape[1:]
    n_actions = env.action_space.shape[0]

    inputs = keras.Input(shape=obs_shape)
    net = layers.Dense(64, activation='relu')(inputs)
    net = layers.Dense(32, activation='relu')(inputs)
    net = layers.Dense(n_actions*2, activation='linear')(net)

    mean, log_stdev = tf.split(net, 2, axis=1)
    log_stdev = tf.clip_by_value(log_stdev, log_stdev_low, log_stdev_high)
    stdev = tf.exp(log_stdev)
    normal = tfp.distributions.Normal(mean, stdev, allow_nan_stats=False)

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


def update_policy(batch, actor, onlines, targets, writer, optimizer, counters, hyp):
    al = hyp['alpha']
    with tf.GradientTape() as tape:
        state_act, log_prob, _ = actor(batch['observation'])
        policy_target = minimum_target(batch['observation'], state_act, targets)
        loss = -1 * (policy_target - al * log_prob)

    grads = tape.gradient(loss, actor.trainable_variables)
    optimizer.apply_gradients(zip(grads, actor.trainable_variables))

    with writer.as_default():
        step = counters['policy_updates']
        tf.summary.scalar('policy target', tf.reduce_mean(policy_target), step=step)
        tf.summary.scalar('policy loss', tf.reduce_mean(loss), step=step)
        tf.summary.scalar('policy logprob * alpha', tf.reduce_mean(al * log_prob), step=step)
        tf.summary.histogram('policy weights', actor.trainable_variables[-2], step=step)
        counters['policy_updates'] += 1
