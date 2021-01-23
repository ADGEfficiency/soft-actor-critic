import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

from sac.utils import minimum_target


#  clip as per stable baselines
log_stdev_low, log_stdev_high = -20, 2
epsilon = 1e-6


def make(env, hyp):
    size_scale = int(hyp['size-scale'])

    obs = env.reset().reshape(1, -1)
    obs_shape = obs.shape[1:]
    n_actions = env.action_space.shape[0]

    inputs = keras.Input(shape=obs_shape)
    net = layers.Dense(64*size_scale, activation='relu')(inputs)
    net = layers.Dense(32*size_scale, activation='relu')(net)
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
    log_prob = tf.reduce_sum(
        log_prob - tf.math.log(1 - action ** 2 + epsilon),
        axis=1,
        keepdims=True
    )

    model = keras.Model(
        inputs=inputs,
        outputs=[action, log_prob, deterministic_action]
    )
    return model


def update(
    batch,
    actor,
    onlines,
    targets,
    log_alpha,
    writer,
    optimizer,
    counters,
):
    al = tf.exp(log_alpha)
    with tf.GradientTape() as tape:
        state_act, log_prob, _ = actor(batch['observation'])
        policy_target = minimum_target(batch['observation'], state_act, targets)
        loss = tf.reduce_mean(al * log_prob - policy_target)

    grads = tape.gradient(loss, actor.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)
    optimizer.apply_gradients(zip(grads, actor.trainable_variables))

    writer.scalar(
        tf.reduce_mean(policy_target),
        'policy-target',
        'policy-updates'
    )
    writer.scalar(
        tf.reduce_mean(loss),
        'policy-loss',
        'policy-updates'
    )
    writer.scalar(
        tf.reduce_mean(log_prob),
        'policy-log-prob',
        'policy-updates'
    )

    counters['policy-updates'] += 1
