import argparse
from collections import defaultdict
import json
from pathlib import Path
import imageio

from sac import checkpoint, policy, utils, json_util
from sac.env import GymWrapper, env_ids
from sac.utils import get_paths, get_latest_run
from sac import utils

import numpy as np



def checkpoint_pipeline(checkpoints):
    out = []
    for path in checkpoints:
        checkpoint = {}
        checkpoint['path'] = path

        checkpoint['results'] = json.loads((path / 'results.json').read_text())
        checkpoint['counters'] = json.loads((path / 'counters.json').read_text())

        out.append(checkpoint)
    return out


def get_best_checkpoint(checkpoints):
    best = np.argmax(checkpoints['episode-reward'])
    return checkpoints['checkpoint'][best]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    args = parser.parse_args()
    run_path = args.run

    checkpoints = checkpoint.load(run_path)
    checkpoints = checkpoint_pipeline(checkpoints)
    checkpoint = get_best_checkpoint(checkpoints)
    print(checkpoint.keys())

    hyp = json_utils.load(Path(run) / 'hyperparameters.json')
    env = GymWrapper('lunar')

    actor = policy.make(env, hyp)
    actor.load_weights(checkpoint / 'actor.h5')

    env.reset()
    obs = env.reset().reshape(1, -1)
    done = False
    episode_reward = 0

    frames = []
    while not done:
        _, _, action = actor(obs)
        frames.append(env.env.render('rgb_array'))
        next_obs, reward, done = env.step(np.array(action))
        episode_reward += reward
        obs = next_obs

    print(episode_reward)

    print('saving gif')
    imageio.mimsave('./render.gif', frames, fps=44)
