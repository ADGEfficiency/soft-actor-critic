from collections import defaultdict
from datetime import datetime

import numpy as np
import tensorflow as tf

from alpha import initialize_log_alpha, update_alpha
from buffer import load_buffer, save_buffer, Buffer
from env import GymWrapper, env_ids
from policy import make_policy, update_policy
from random_policy import make_random_policy
from qfunc import update_target_networks, initialize_qfuncs, update_qfuncs
from utils import *


def episode(
    env,
    buffer,
    policy,
    hyp,
    counters=None,
    logger=None,
    mode='train'
):
    obs = env.reset().reshape(1, -1)
    done = False

    reward_scale = hyp['reward-scale']
    episode_rewards = []

    while not done:
        action, _, deterministic_action = policy(obs)

        if mode == 'test':
            action = deterministic_action

        next_obs, reward, done = env.step(np.array(action))
        buffer.append(env.Transition(obs, action, reward/reward_scale, next_obs, done))
        episode_rewards.append(reward)

        if logger:
            logger.debug(
                f'{obs}, {action}, {reward}, {next_obs}, {done}, {mode}'
            )

        counters['env-steps'] += 1
        obs = next_obs

    counters['episodes'] += 1
    counters[f'{mode}-episodes'] += 1

    return buffer, episode_rewards


def last_100_episode_rewards(rewards):
    last = rewards[-100:]
    return sum(last) / len(last)


def fill_buffer_random_policy(
    env,
    buffer,
    hyp,
    writer,
    counters,
    logger,
):
    print(f"filling buffer with {buffer.size} samples")
    random_policy = make_random_policy(env)

    while not buffer.full:
        buffer, _ = episode(
            env,
            buffer,
            random_policy,
            hyp,
            writer=writer,
            counters=counters,
            logger=logger,
            mode='random'
        )
    assert len(buffer) == buffer.size
    print(f"buffer filled with {len(buffer)} samples")
    return buffer




def train(
    batch,
    actor,
    onlines,
    targets,
    log_alpha,
    writer,
    qfunc_optimizers,
    pol_optimizer,
    counters,
    hyp
):
    update_qfuncs(
        batch,
        actor,
        onlines,
        targets,
        log_alpha,
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
        log_alpha,
        writer,
        pol_optimizer,
        counters
    )

    update_target_networks(
        onlines,
        targets,
        hyp,
        counters
    )

    update_alpha(
        batch,
        actor,
        log_alpha,
        target_entropy,
        alpha_optimizer,
        counters,
        writer
    )


def test(
    env,
    buffer,
    actor,
    writer,
    counters,
    hyp,
    logger,
):
    test_results = []
    for _ in range(hyp['n-tests']):
        buffer, test_episode_reward = episode(
            env,
            buffer,
            actor,
            hyp,
            counters,
            logger,
            mode='test'
        )
        test_results.append(sum(test_episode_reward))

    return test_results


if __name__ == '__main__':
    hyp = {
        'initial-log-alpha': 0.0,
        'gamma': 0.99,
        'rho': 0.995,
        'buffer-size': int(1e6),
        'reward-scale': 5,
        'lr': 3e-4,
        'batch-size': 1024,
        'episodes': 500000,
        'updates': 5,
        'test-every': 1,
        'n-tests': 10,
        'size_scale': 6,
        'time': datetime.utcnow().isoformat()
    }

    n_episodes = int(hyp['episodes'])
    n_updates = int(hyp['updates'])

    counters = defaultdict(int)
    writers = {
        'random': Writer('random', counters),
        'train': Writer('train', counters),
        'test': Writer('test', counters)
    }

    transition_logger = make_logger('transitions.data')

    env = GymWrapper(env_ids['lunar'])

    actor = make_policy(env, size_scale=hyp['size_scale'])

    onlines, targets = initialize_qfuncs(env, size_scale=hyp['size_scale'])

    target_entropy, log_alpha = initialize_log_alpha(
        env,
        initial_value=hyp['initial-log-alpha']
    )

    hyp['target-entropy'] = float(target_entropy)

    from utils import PATH
    dump_json(hyp, PATH / 'hyperparameters.json')


    # if bpath.exists():
    if True:
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
        save_buffer(buffer, 'random')

    qfunc_optimizers = [
        tf.keras.optimizers.Adam(learning_rate=hyp['lr'])
        for _ in range(2)
    ]
    pol_optimizer = tf.keras.optimizers.Adam(learning_rate=hyp['lr'])
    alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=hyp['lr'])

    rewards = []
    for _ in range(n_episodes):
        buffer, train_rewards = episode(
            env,
            buffer,
            actor,
            hyp,
            counters,
            logger=transition_logger,
            mode='train',
        )

        writers['train'].scalar(
            np.sum(train_rewards),
            'episode-reward',
            'episodes'
        )

        rewards.append(sum(train_rewards))
        writers['train'].scalar(
            last_100_episode_rewards(rewards),
            'last-100-episode-rewards',
            'episodes'
        )
        print('train', counters['train-episodes'], np.sum(train_rewards))

        print(f'training for {len(train_rewards)} steps')
        for _ in range(len(train_rewards)):
            batch = buffer.sample(hyp['batch-size'])
            train(
                batch,
                actor,
                onlines,
                targets,
                log_alpha,
                writers['train'],
                qfunc_optimizers,
                pol_optimizer,
                counters,
                hyp
            )

        if counters['episodes'] % hyp['test-every'] == 0:
            test_rewards = test(
                env,
                buffer,
                actor,
                writers['test'],
                counters,
                hyp,
                logger=transition_logger
            )
            test_rewards = np.mean(test_rewards)

            dump(
                actor,
                episode=counters['episodes'],
                reward=np.mean(test_rewards)
            )

            writers['test'].scalar(
                test_rewards,
                'test-episode-rewards',
                'test-episodes'
            )
            print('test', counters['test-episodes'], test_rewards)
