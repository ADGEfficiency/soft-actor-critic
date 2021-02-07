from collections import defaultdict
import random
from time import sleep
import time

import click
import numpy as np
import tensorflow as tf

from sac import alpha, checkpoint, json_util
from sac import alpha, memory, policy, qfunc, random_policy, target, utils
from sac.env import GymWrapper



def now():
    return time.perf_counter()


def init_nets(env, hyp):
    actor = policy.make(env, hyp)
    onlines, targets = qfunc.make(env, size_scale=hyp['size-scale'])
    target_entropy, log_alpha = alpha.make(env, initial_value=hyp['initial-log-alpha'])
    return {
        'actor': actor,
        'online-1': onlines[0],
        'online-2': onlines[1],
        'target-1': targets[0],
        'target-2': targets[1],
        'target_entropy': float(target_entropy),
        'alpha': log_alpha,
    }


def init_writers(counters, paths):
    return {
        'random': utils.Writer('random', counters, paths['run']),
        'test': utils.Writer('test', counters, paths['run']),
        'train': utils.Writer('train', counters, paths['run'])
    }


def init_optimizers(hyp):
    return {
        'online-1': tf.keras.optimizers.Adam(learning_rate=hyp['lr']),
        'online-2': tf.keras.optimizers.Adam(learning_rate=hyp['lr']),
        'actor': tf.keras.optimizers.Adam(learning_rate=hyp['lr']),
        'alpha': tf.keras.optimizers.Adam(learning_rate=hyp['lr']),
    }


def init_fresh(hyp):
    counters = defaultdict(int)
    paths = utils.get_paths(hyp)

    env = GymWrapper(hyp['env-name'])
    buffer = memory.make(env, hyp)

    nets = init_nets(env, hyp)
    writers = init_writers(counters, paths)
    optimizers = init_optimizers(hyp)

    transition_logger = utils.make_logger('transitions.data', paths['run'])

    target_entropy = nets.pop('target_entropy')
    hyp['target-entropy'] = target_entropy

    rewards = defaultdict(list)
    return {
        'hyp': hyp,
        'paths': paths,
        'counters': counters,
        'env': env,
        'buffer': buffer,
        'nets': nets,
        'writers': writers,
        'optimizers': optimizers,
        'transition_logger': transition_logger,
        'rewards': rewards
    }


def init_checkpoint(checkpoint_path):
    point = checkpoint.load_checkpoint(checkpoint_path)
    hyp = point['hyp']
    paths = utils.get_paths(hyp)
    counters = point['counters']

    writers = init_writers(counters, paths)

    transition_logger = utils.make_logger('transitions.data', paths['run'])
    c = point

    rewards = point['rewards']
    return {
        'hyp': hyp,
        'paths': paths,
        'counters': counters,
        'env': c['env'],
        'buffer': c['buffer'],
        'nets': c['nets'],
        'writers': writers,
        'optimizers': c['optimizers'],
        'transition_logger': transition_logger,
        'rewards': rewards
    }


def run_episode(
    env,
    buffer,
    actor,
    hyp,
    writers,
    counters,
    rewards,
    mode,
    logger=None
):
    episode_rewards = episode(
        env,
        buffer,
        actor,
        hyp,
        writers,
        counters,
        rewards,
        mode,
        logger=logger
    )
    episode_reward = float(sum(episode_rewards))

    rewards['episode-reward'].append(episode_reward)
    rewards[f'{mode}-reward'].append(episode_reward)

    writers[mode].scalar(
        episode_reward,
        f'{mode}-episode-reward',
        f'{mode}-episodes',
        verbose=True
    )
    writers[mode].scalar(
        episode_reward,
        'episode-reward',
        'episodes'
    )
    return episode_rewards


def episode(
    env,
    buffer,
    actor,
    hyp,
    writers,
    counters,
    rewards,
    mode,
    logger=None
):
    obs = env.reset().reshape(1, -1)
    done = False

    reward_scale = hyp['reward-scale']
    episode_rewards = []

    while not done:
        action, _, deterministic_action = actor(obs)

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

    return episode_rewards


def sample_random(
    env,
    buffer,
    hyp,
    writers,
    counters,
    rewards,
    logger,
):
    mode = 'random'
    print(f"filling buffer with {buffer.size} samples")
    policy = random_policy.make(env)

    while not buffer.full:
        run_episode(
            env,
            buffer,
            policy,
            hyp,
            writers,
            counters,
            rewards,
            mode,
            logger=logger
        )

    assert len(buffer) == buffer.size
    print(f"buffer filled with {len(buffer)} samples\n")
    return buffer


def sample_test(
    env,
    buffer,
    actor,
    hyp,
    writers,
    counters,
    rewards,
    logger,
):
    mode = 'test'
    test_results = []
    for _ in range(hyp['n-tests']):

        test_rewards = run_episode(
            env,
            buffer,
            actor,
            hyp,
            writers,
            counters,
            rewards,
            mode,
            logger=logger
        )
        test_results.append(sum(test_rewards))

    return test_results


def sample_train(
    env,
    buffer,
    actor,
    hyp,
    writers,
    counters,
    rewards,
    logger,
):
    mode = 'train'
    return run_episode(
        env,
        buffer,
        actor,
        hyp,
        writers,
        counters,
        rewards,
        mode,
        logger=logger
    )


def train(
    batch,
    actor,
    onlines,
    targets,
    log_alpha,
    writer,
    optimizers,
    counters,
    hyp
):
    st = now()
    qfunc.update(
        batch,
        actor,
        onlines,
        targets,
        log_alpha,
        writer,
        [optimizers['online-1'], optimizers['online-2']],
        counters,
        hyp
    )
    counters['q-func-update-seconds'] = now() - st

    st = now()
    policy.update(
        batch,
        actor,
        onlines,
        targets,
        log_alpha,
        writer,
        optimizers['actor'],
        counters
    )
    counters['pol-func-update-seconds'] = now() - st

    st = now()
    target.update(
        onlines,
        targets,
        hyp,
        counters
    )
    counters['target-update-seconds'] = now() - st

    st = now()
    alpha.update(
        batch,
        actor,
        log_alpha,
        hyp,
        optimizers['alpha'],
        counters,
        writer
    )
    counters['alpha-update-seconds'] = now() - st

    counters['train-steps'] += 1


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.seed(seed)


def main(
    hyp,
    paths,
    counters,
    env,
    buffer,
    nets,
    writers,
    optimizers,
    transition_logger,
    rewards
):
    json_util.save(hyp, paths['run'] / 'hyperparameters.json')

    if not buffer.full:
        sample_random(
            env,
            buffer,
            hyp,
            writers,
            counters,
            rewards,
            transition_logger,
        )
        memory.save(buffer, paths['run'] / 'random.pkl')
        memory.save(buffer, paths['experiment'] / 'random.pkl')

    rewards = defaultdict(list)
    for _ in range(int(hyp['n-episodes'])):
        if counters['train-episodes'] % hyp['test-every'] == 0:
            test_rewards = sample_test(
                env,
                buffer,
                nets['actor'],
                hyp,
                writers,
                counters,
                rewards,
                transition_logger
            )

            checkpoint.save(
                hyp,
                nets,
                optimizers,
                buffer,
                episode=counters['test-episodes'],
                rewards=rewards,
                counters=counters,
                paths=paths
            )

        writers['train'].scalar(
            utils.last_100_episode_rewards(rewards['episode-reward']),
            'last-100-episode-rewards',
            'episodes'
        )

        train_rewards = sample_train(
            env,
            buffer,
            nets['actor'],
            hyp,
            writers,
            counters,
            rewards,
            transition_logger
        )
        train_steps = len(train_rewards)

        print(f'training \n step {counters["train-steps"]:6.0f}, {train_steps} steps')
        for _ in range(train_steps):
            batch = buffer.sample(hyp['batch-size'])
            train(
                batch,
                nets['actor'],
                [nets['online-1'], nets['online-2']],
                [nets['target-1'], nets['target-2']],
                nets['alpha'],
                writers['train'],
                optimizers,
                counters,
                hyp
            )

    if counters['train-episodes'] % hyp['test-every'] == 0:
        test_rewards = sample_test(
            env,
            buffer,
            nets['actor'],
            hyp,
            writers,
            counters,
            rewards,
            transition_logger
        )

        checkpoint.save(
            hyp,
            nets,
            optimizers,
            buffer,
            episode=counters['test-episodes'],
            rewards=rewards,
            counters=counters,
            paths=paths
        )


@click.command()
@click.argument("experiment-json", nargs=1)
@click.option("-n", "--run-name", default=None)
@click.option("-b", "--buffer", nargs=1, default="new")
@click.option("-s", "--seed", nargs=1, default=None)
@click.option("-c", "--checkpoint_path", nargs=1, default=None)
def cli(experiment_json, run_name, buffer, seed, checkpoint_path):

    print('cli')
    print('------')
    print(experiment_json, run_name, buffer)
    print('')

    hyp = json_util.load(experiment_json)
    hyp['buffer'] = buffer

    if run_name:
        hyp['run-name'] = run_name

    if not seed:
        from random import choice
        seed = choice(range(int(1e4)))

    hyp['seed'] = seed
    print('params')
    print('------')
    print(hyp)
    print('')
    sleep(2)

    if checkpoint_path:
        print(f'checkpointing from {checkpoint_path}')
        print('')
        main(**init_checkpoint(checkpoint_path))

    else:
        print(f'starting so fresh, so clean')
        print('')
        main(**init_fresh(hyp))


if __name__ == '__main__':
    cli()
