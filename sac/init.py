from collections import defaultdict
import tensorflow as tf

from sac import utils, memory, policy, qfunc, alpha
from sac.env import GymWrapper



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


