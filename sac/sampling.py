import numpy as np

from sac import random_policy


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
