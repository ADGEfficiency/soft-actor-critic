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

    update_params(online=q1, target=q1_target, rho=0.0)
    update_params(online=q2, target=q2_target, rho=0.0)
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
    rho = 0.99

    target = batch['reward'] + ga * (1 - d) * (next_state_target - al * log_prob)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for onl in onlines:
        with tf.GradientTape() as tape:
            loss = tf.keras.losses.MSE(
                onl([batch['observation'], batch['action']]), target
            )

        grads = tape.gradient(loss, onl.trainable_variables)
        optimizer.apply_gradients(zip(grads, onl.trainable_variables))

    #  update policy
    with tf.GradientTape() as tape:
        pol_act, log_prob, _ = policy(batch['observation'])
        state_target = tf.reduce_min([
            t([batch['observation'], pol_act]) for t in targets
        ], axis=0)
        loss = -1 * tf.reduce_mean(state_target - al * log_prob)

    grads = tape.gradient(loss, policy.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))

    #  update target networks
    for onl, tar in zip(onlines, targets):
        update_params(onl, tar, rho=rho)

    with writer.as_default():
        tf.summary.scalar('loss', loss, step=1)
        tf.summary.histogram('policy-weights', policy.trainable_variables[-2], step=1)
