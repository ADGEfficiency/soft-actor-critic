import gym
import tensorflow as tf

from buffer import Buffer
from env import env, elements, Transition
from policy import make_policy


def episode(env, buffer):
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        buffer.append(Transition(obs, action, reward, next_obs, done))
    return buffer


if __name__ == '__main__':
    writer = tf.summary.create_file_writer('./logs')


    buffer = Buffer(elements, size=64)
    buffer = episode(env, buffer)
    batch = buffer.sample(64)

    #  setup policy
    policy = make_policy(env)

    #  setup qfuncs
    obs = env.reset().reshape(1, -1)
    assert obs.ndim == 2
    obs_shape = (obs.shape[1], )
    n_actions = (env.action_space.shape[0], )

    from qfunc import make_qfunc, update_params
    q1 = make_qfunc(obs_shape, n_actions)
    q1_target = make_qfunc(obs_shape, n_actions)

    q2 = make_qfunc(obs_shape, n_actions)
    q2_target = make_qfunc(obs_shape, n_actions)

    q1, q1_target = update_params(online=q1, target=q1_target, rho=0.0)
    q2, q2_target = update_params(online=q2, target=q2_target, rho=0.0)
    onlines = [q1, q2]
    targets = [q1_target, q2_target]

    #  computing targets for q funcs

    pol_act, log_prob, _ = policy(batch['next_observation'])
    next_state_target = tf.reduce_min([
        t([batch['next_observation'], pol_act])for t in targets
    ], axis=0)

    d = 1.0
    al = 1.0  #  alpha
    ga = 1.0  #  gamma

    target = batch['reward'] + ga * (1 - d) * (next_state_target - al * log_prob)

    with tf.GradientTape() as tape:
        for 

    grads = tape.gradient(loss, q1.trainable_variables)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer.apply_gradients(zip(grads, q1.trainable_variables))


    #  update policy
    with tf.GradientTape() as tape:
        _, log_probs, _ = policy(batch['observation'])
        loss = -1 * tf.reduce_mean(q1_error - log_probs)

    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))

    #  update target networks
    # TODO
    p = 0.99
    for online, target in zip(q1.trainable_variables, q1_target.trainable_variables):
        target.assign(p * target.value() + (1 - p) * online.value())

    with writer.as_default():
        tf.summary.scalar('loss', loss, step=1)
        tf.summary.histogram('policy-weights', policy.trainable_variables[-2], step=1)
