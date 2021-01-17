from collections import defaultdict

import numpy as np
import tensorflow as tf

from buffer import bpath, load_buffer, save_buffer, Buffer
from env import GymWrapper, env_ids
from policy import make_policy, update_policy
from random_policy import make_random_policy
from qfunc import update_target_network, initialize_qfuncs, update_qfuncs
from logger import make_logger
from utils import dump_json


def episode(
    env,
    buffer,
    policy,
    writer=None,
    counters=None,
    reward_scale=1.0,
    logger=None,
    mode='train'
):
    obs = env.reset().reshape(1, -1)
    done = False
    episode_reward = 0
    while not done:
        action, _, deterministic_action = policy(obs)

        if mode == 'test':
            action = deterministic_action

        next_obs, reward, done = env.step(np.array(action))
        buffer.append(env.Transition(obs, action, reward/reward_scale, next_obs, done))
        episode_reward += reward

        if writer:
            with writer.as_default():
                step = counters['global-env-steps']
                tf.summary.scalar('scaled action', action[0][0], step=step)
                counters['global-env-steps'] += 1

        if logger:
            logger.debug(
                f'{obs}, {action}, {reward}, {next_obs}, {done}, {mode}'
            )

        obs = next_obs

    if writer:
        with writer.as_default():
            step = counters['episodes']
            tf.summary.scalar('episode reward', tf.reduce_mean(episode_reward), step=step)
            counters['episodes'] += 1
            print(mode, step, episode_reward)
    return buffer, episode_reward


def last_100_episode_rewards(rewards):
    last = rewards[-100:]
    return sum(last) / len(last)


def fill_buffer_random_policy(
    env,
    buffer,
    reward_scale=1.0,
    writer=None,
    counters=None,
    logger=None,
):
    print(f"filling buffer with {buffer.size} samples")
    random_policy = make_random_policy(env)
    while not buffer.full:
        buffer, _ = episode(
            env,
            buffer,
            random_policy,
            counters=counters,
            reward_scale=reward_scale,
            logger=logger,
            writer=writer,
            mode='random'
        )
    assert len(buffer) == buffer.size
    print(f"buffer filled with {len(buffer)} samples")
    return buffer


if __name__ == '__main__':
    from datetime import datetime
    hyp = {
        'alpha': 0.3,
        'gamma': 0.99,
        'rho': 0.995,
        'buffer-size': int(1e6),
        'reward-scale': 10,
        'lr': 0.001,
        'episodes': 25000,
        'updates': 5,
        'test-every': 50,
        'n_tests': 10,
        'size_scale': 6,
        'time': datetime.utcnow().isoformat()
    }

    n_episodes = int(hyp['episodes'])
    n_updates = int(hyp['updates'])

    writer = tf.summary.create_file_writer('./logs/sac')
    random_writer = tf.summary.create_file_writer('./logs/random')
    dump_json(hyp, './logs/hyperparameters.json')
    transition_logger = make_logger('./logs/transitions.data')

    env = GymWrapper(env_ids['lunar'])

    actor = make_policy(env, size_scale=hyp['size_scale'])

    onlines, targets = initialize_qfuncs(env, size_scale=hyp['size_scale'])

    counters = defaultdict(int)

    if bpath.exists():
        print('loaded buffer')
        buffer = load_buffer()
        assert buffer.full
    else:
        buffer = Buffer(env.elements, size=hyp['buffer-size'])
        buffer = fill_buffer_random_policy(
            env,
            buffer,
            reward_scale=hyp['reward-scale'],
            counters=counters,
            writer=random_writer,
            logger=transition_logger
        )
        save_buffer(buffer)

    qfunc_optimizers = [
        tf.keras.optimizers.Adam(learning_rate=hyp['lr'])
        for _ in range(2)
    ]
    pol_optimizer = tf.keras.optimizers.Adam(learning_rate=hyp['lr'])

    rewards = []
    for _ in range(n_episodes):
        buffer, episode_reward = episode(
            env,
            buffer,
            actor,
            writer,
            counters,
            reward_scale=hyp['reward-scale'],
            logger=transition_logger,
            mode='train',
        )

        rewards.append(episode_reward)

        with writer.as_default():
            tf.summary.scalar(
                'last 100 episode reward',
                last_100_episode_rewards(rewards),
                counters['episodes']
            )

        for _ in range(n_updates):
            batch = buffer.sample(1024)

            update_qfuncs(
                batch,
                actor,
                onlines,
                targets,
                writer,
                qfunc_optimizers,
                counters,
                hyp
            )

            update_policy(
                batch,
                actor,
                onlines,
                targets,
                writer,
                pol_optimizer,
                counters,
                hyp
            )

            #  update target networks
            for onl, tar in zip(onlines, targets):
                update_target_network(
                    onl,
                    tar,
                    hyp['rho'],
                    step=counters['qfunc_updates'],
                    writer=writer
                )

        if counters['episodes'] % hyp['test-every'] == 0:
            test_results = []
            for _ in range(hyp['n_tests']):
                buffer, test_episode_reward = episode(
                    env,
                    buffer,
                    actor,
                    reward_scale=hyp['reward-scale'],
                    logger=transition_logger,
                    mode='test'
                )
                test_results.append(test_episode_reward)

            test_results = np.mean(test_results)
            with writer.as_default():
                counters['test-episodes'] += 1
                step = counters['test-episodes']
                tf.summary.scalar('test episode rewards', np.mean(test_results), step=step)
                print('test', step, test_results)
                actor.save_weights("logs/pol.h5")
