from collections import defaultdict
from time import sleep

import numpy as np
import tensorflow as tf
import click

from sac import alpha, checkpoint, json_util
from sac.env import GymWrapper

from sac import alpha, memory, policy, qfunc, random_policy, target, utils


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
    writers,
    counters,
    logger,
):
    print(f"filling buffer with {buffer.size} samples")
    policy = random_policy.make(env)

    while not buffer.full:
        buffer, random_episode_reward  = episode(
            env,
            buffer,
            policy,
            hyp,
            counters=counters,
            logger=logger,
            mode='random'
        )
        random_episode_reward = sum(random_episode_reward)

        writers['random'].scalar(
            random_episode_reward,
            'random-episode-reward',
            'random-episodes',
            verbose=True
        )
        writers['random'].scalar(
            random_episode_reward,
            'episode-reward',
            'episodes'
        )

    assert len(buffer) == buffer.size
    print(f"buffer filled with {len(buffer)} samples\n")
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
    alpha_optimizer,
    counters,
    hyp
):
    qfunc.update(
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

    policy.update(
        batch,
        actor,
        onlines,
        targets,
        log_alpha,
        writer,
        pol_optimizer,
        counters
    )

    target.update(
        onlines,
        targets,
        hyp,
        counters
    )

    alpha.update(
        batch,
        actor,
        log_alpha,
        hyp,
        alpha_optimizer,
        counters,
        writer
    )
    counters['train-steps'] += 1


def test(
    env,
    buffer,
    actor,
    writers,
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
        test_episode_reward = float(sum(test_episode_reward))
        test_results.append(test_episode_reward)

        writers['test'].scalar(
            test_episode_reward,
            'test-episode-reward',
            'test-episodes',
            verbose=True
        )
        writers['test'].scalar(
            test_episode_reward,
            'episode-reward',
            'episodes'
        )

    return test_results


def main(hyp):
    counters = defaultdict(int)

    paths = utils.get_paths(hyp)

    env = GymWrapper(hyp['env-name'])
    buffer = memory.make(env, hyp)

    writers = {
        'random': utils.Writer('random', counters, paths['run']),
        'test': utils.Writer('test', counters, paths['run']),
        'train': utils.Writer('train', counters, paths['run'])
    }
    transition_logger = utils.make_logger('transitions.data', paths['run'])

    actor = policy.make(env, hyp)
    onlines, targets = qfunc.make(env, size_scale=hyp['size-scale'])
    target_entropy, log_alpha = alpha.make(env, initial_value=hyp['initial-log-alpha'])

    hyp['target-entropy'] = float(target_entropy)
    json_util.save(hyp, paths['run'] / 'hyperparameters.json')

    if not buffer.full:
        buffer = fill_buffer_random_policy(
            env,
            buffer,
            hyp,
            writers=writers,
            counters=counters,
            logger=transition_logger
        )
        memory.save(buffer, paths['run'] / 'random.pkl')
        memory.save(buffer, paths['experiment'] / 'random.pkl')

    qfunc_optimizers = [
        tf.keras.optimizers.Adam(learning_rate=hyp['lr'])
        for _ in range(2)
    ]
    pol_optimizer = tf.keras.optimizers.Adam(learning_rate=hyp['lr'])
    alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=hyp['lr'])

    rewards = []
    for _ in range(int(hyp['n-episodes'])):
        if counters['train-episodes'] % hyp['test-every'] == 0:
            test_rewards = test(
                env,
                buffer,
                actor,
                writers,
                counters,
                hyp,
                logger=transition_logger
            )

            checkpoint.save(
                actor,
                episode=counters['test-episodes'],
                rewards=test_rewards,
                paths=paths
            )

        buffer, train_rewards = episode(
            env,
            buffer,
            actor,
            hyp,
            counters,
            logger=transition_logger,
            mode='train',
        )
        print('')
        train_steps = len(train_rewards)

        train_rewards = sum(train_rewards)
        rewards.append(train_rewards)

        writers['train'].scalar(
            train_rewards,
            'train-episode-reward',
            'train-episodes',
            verbose=True
        )
        writers['train'].scalar(
            train_rewards,
            'episode-reward',
            'episodes'
        )

        writers['train'].scalar(
            last_100_episode_rewards(rewards),
            'last-100-episode-rewards',
            'episodes'
        )

        print(f'training \n step {counters["train-steps"]:6.0f}, {train_steps} steps')
        for _ in range(train_steps):
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
                alpha_optimizer,
                counters,
                hyp
            )

    if counters['train-episodes'] % hyp['test-every'] == 0:
        test_rewards = test(
            env,
            buffer,
            actor,
            writers,
            counters,
            hyp,
            logger=transition_logger
        )

        checkpoint.save(
            actor,
            episode=counters['test-episodes'],
            rewards=test_rewards,
            paths=paths
        )


@click.command()
@click.argument("experiment-json", nargs=1)
@click.option("-n", "--name", default=None)
@click.option("-b", "--buffer", nargs=1, default="new")
@click.option("-s", "--seed", nargs=1, default=None)
def cli(experiment_json, name, buffer, seed):

    print('cli')
    print('------')
    print(experiment_json, name, buffer)
    print('')

    hyp = json.load(experiment_json)
    hyp['buffer'] = buffer

    if name:
        hyp['run-name'] = name

    if not seed:
        from random import choice
        seed = choice(range(int(1e4)))

    hyp['seed'] = seed
    print('params')
    print('------')
    print(hyp)
    print('')

    sleep(2)
    main(hyp)


if __name__ == '__main__':
    cli()
