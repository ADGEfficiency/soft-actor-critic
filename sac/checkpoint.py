from datetime import datetime
from collections import defaultdict
from pathlib import Path
import pickle
import tensorflow as tf

import numpy as np

from sac import json_util, memory
from sac.env import GymWrapper


def save(
    hyp,
    nets,
    optimizers,
    buffer,
    episode,
    rewards,
    counters,
    paths
):
    path = paths['run'] / 'checkpoints' / f'test-episode-{episode}'
    path.mkdir(exist_ok=True, parents=True)

    for name, net in nets.items():
        if 'alpha' not in name:
            net.save_weights(path / f'{name}.h5')

    #  save alpha!
    log_alpha = nets['alpha'].numpy()
    np.save(path / 'alpha.npy', log_alpha)

    for name, optimizer in optimizers.items():
        wts = optimizer.get_weights()
        if wts:
            opt_path = path / f'{name}.pkl'
            with opt_path.open('wb') as fi:
                pickle.dump(wts, fi)

    memory.save(buffer, path / 'buffer.pkl')

    rewards = dict(rewards)
    rewards['time'] = datetime.utcnow().isoformat()
    json_util.save(
        rewards,
        path / 'rewards.json'
    )
    json_util.save(
        dict(counters),
        path / 'counters.json'
    )
    json_util.save(
        hyp,
        path / 'hyperparameters.json'
    )


def load(run):
    checkpoints = Path(run) / 'checkpoints'
    checkpoints = [
        load_checkpoint(p) for p in checkpoints.iterdir() if p.is_dir()
    ]
    return checkpoints


def load_checkpoint(path):

    hyp = json_util.load(path / 'hyperparameters.json')
    env = GymWrapper(hyp['env-name'])

    from sac.main import init_nets, init_optimizers
    nets = init_nets(env, hyp)
    #  awkward
    nets.pop('target_entropy')
    for name, net in nets.items():
        #  awkward
        if 'alpha' not in name:
            net.load_weights(path / f'{name}.h5')

    log_alpha = nets['alpha']
    saved_log_alpha = np.load(path / 'alpha.npy')
    log_alpha.assign(saved_log_alpha)

    optimizers = init_optimizers(hyp)
    for name, opt in optimizers.items():
        opt_path = path / f'{name}.pkl'

        if opt_path.exists():
            # https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state
            model = nets[name]
            #  single var
            if 'alpha' in name:
                wts = [model, ]
            else:
                wts = model.trainable_variables
            zero_grads = [tf.zeros_like(w) for w in wts]
            opt.apply_gradients(zip(zero_grads, wts))

            with opt_path.open('rb') as fi:
                opt.set_weights(pickle.load(fi))

    buffer = memory.load(path / 'buffer.pkl')

    rewards = json_util.load(path / 'rewards.json')
    rewards.pop('time')
    rewards = defaultdict(list, rewards)

    counters = json_util.load(path / 'counters.json')
    counters = defaultdict(int, counters)
    return {
        'path': path,
        'hyp': hyp,
        'env': env,
        'nets': nets,
        'optimizers': optimizers,
        'buffer': buffer,
        'rewards': rewards,
        'counters': counters,
    }
