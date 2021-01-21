import json
import logging

from pathlib import Path

import tensorflow as tf

def minimum_target(state, action, targets):
    return tf.reduce_min([t([state, action]) for t in targets], axis=0)

PATH = Path('./experiments/env_id/run_id')
PATH.mkdir(exist_ok=True, parents=True)


def dump_json(data, file):
    file = str(file)
    with open(file, 'w') as fi:
        json.dump(data, fi)




def dump(
    actor,
    episode,
    reward
):
    path = PATH / 'episodes' / str(episode)
    path.mkdir(exist_ok=True, parents=True)
    actor.save_weights(path / 'actor.h5')
    dump_json(
        {'episode': episode, 'reward': reward},
        path / 'results.json'
    )




def make_logger(log_file):
    """info to STDOUT, debug to file"""
    level = logging.DEBUG

    # Create a custom logger
    fldr = PATH / 'logs'
    fldr.mkdir(exist_ok=True, parents=True)
    logger = logging.getLogger(log_file)
    logger.setLevel(level)

    # Create handlers
    c_handler = logging.StreamHandler()
    if log_file:
        f_handler = logging.FileHandler(str(fldr / log_file))
        f_format = logging.Formatter("%(asctime)s, %(name)s, %(levelname)s, %(message)s")
        f_handler.setFormatter(f_format)
        f_handler.setLevel(logging.DEBUG)
        logger.addHandler(f_handler)

    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter("%(name)s, %(levelname)s, %(message)s")
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)
    return logger


import numpy as np


class Writer:
    def __init__(self, path, counters):
        path = PATH / 'tensorboard-logs' / path
        self.writer = tf.summary.create_file_writer(str(path))
        self.counters = counters

    def scalar(self, value, name, counter, verbose=False):
        value = np.array(value)

        with self.writer.as_default():
            step = self.counters[counter]
            tf.summary.scalar(name, np.mean(value), step=step)

        if verbose:
            print(f'{name} \n step {self.counters[counter]:6.0f}, mu {np.mean(value):4.2f}, sig {np.std(value):4.2f}')
