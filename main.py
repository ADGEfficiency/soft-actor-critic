import gym
import tensorflow as tf

from buffer import Buffer
from env import GymWrapper
from policy import make_policy, make_random_policy
from qfunc import make_qfunc, update_target_network


def episode(
    env,
    buffer,
    policy,
    writer=None,
    counters=None,
    reward_scale=1.0
):
    obs = env.reset().reshape(1, -1)
    done = False
    episode_reward = 0
    while not done:
        action, _, _ = policy(obs)
        next_obs, reward, done = env.step(action)
        buffer.append(env.Transition(obs, action, reward/reward_scale, next_obs, done))
        episode_reward += reward
        if writer:
            with writer.as_default():
                step = counters['global-env-steps']
                tf.summary.scalar('action', action[0][0], step=step)
                counters['global-env-steps'] += 1

    if writer:
        with writer.as_default():
            step = counters['episodes']
            tf.summary.scalar('episode reward', tf.reduce_mean(episode_reward), step=step)
            counters['episodes'] += 1
            print(step, episode_reward)
    return buffer, episode_reward


def last_100_episode_rewards(rewards):
    last = rewards[-100:]
    return sum(last)[0] / len(last)


def fill_buffer_random_policy(env, buffer, reward_scale):
    random_policy = make_random_policy(env)
    while not buffer.full:
        buffer, _ = episode(env, buffer, random_policy, reward_scale=reward_scale)
    assert len(buffer) == buffer.size
    return buffer


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


def minimum_target(state, action, targets):
    return tf.reduce_min([t([state, action]) for t in targets], axis=0)


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


def update_policy(batch, actor, onlines, targets, writer, optimizer, counters, hyp):
    al = hyp['alpha']
    with tf.GradientTape() as tape:
        state_act, log_prob, _ = actor(batch['observation'])
        state_target = minimum_target(batch['observation'], state_act, targets)
        loss = -1 * (state_target - al * log_prob)

    grads = tape.gradient(loss, actor.trainable_variables)
    optimizer.apply_gradients(zip(grads, actor.trainable_variables))

    with writer.as_default():
        step = counters['policy_updates']
        tf.summary.scalar('policy target', tf.reduce_mean(state_target), step=step)
        tf.summary.scalar('policy loss', tf.reduce_mean(loss), step=step)
        tf.summary.histogram('policy weights', actor.trainable_variables[-2], step=step)
        counters['policy_updates'] += 1


if __name__ == '__main__':
    hyp = {'alpha': 1.0/5.0, 'gamma': 0.9, 'rho': 0.95}
    writer = tf.summary.create_file_writer('./logs')

    env = GymWrapper('Pendulum-v0')
    buffer = Buffer(env.elements, size=int(1e6))

    #  create our agent policy (the actor)
    actor = make_policy(env)

    #  create our qfuncs
    onlines, targets = initialize_qfuncs(env)

    #  fill the buffer
    buffer = fill_buffer_random_policy(env, buffer, reward_scale=100)

    n_episodes = 20000
    n_updates = 16  # TODO

    from collections import defaultdict
    counters = defaultdict(int)

    qfunc_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

    rewards = []
    for _ in range(n_episodes):
        buffer, episode_reward = episode(
            env,
            buffer,
            actor,
            writer,
            counters,
            reward_scale=100
        )
        rewards.append(episode_reward)
        with writer.as_default():
            tf.summary.scalar(
                'last 100 episode reward',
                last_100_episode_rewards(rewards),
                counters['episodes']
            )

        for _ in range(n_updates):
            batch = buffer.sample(256)

            update_qfuncs(
                batch,
                actor,
                onlines,
                targets,
                writer,
                qfunc_optimizer,
                counters,
                hyp
            )

            update_policy(
                batch,
                actor,
                onlines,
                targets,
                writer,
                qfunc_optimizer,
                counters,
                hyp
            )

    #  update target networks
    for onl, tar in zip(onlines, targets):
        update_target_network(onl, tar, hyp['rho'])

