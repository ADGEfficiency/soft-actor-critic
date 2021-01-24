import json
import logging
import os
from pathlib import Path

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def minimum_target(state, action, targets):
    return tf.reduce_min([t([state, action]) for t in targets], axis=0)


def get_latest_run(experiment):
    runs = [p.name for p in experiment.iterdir() if p.is_dir()]
    runs = [run.split('-')[-1] for run in runs]
    runs = [run for run in runs if run.isdigit()]

    if runs:
        return max(runs)
    else:
        return 0


def get_paths(hyp):
    #  experiments/results/EXPTNAME/RUNNAME
    results = Path.cwd() / 'experiments' / 'results'
    experiment = results / hyp['env-name']
    experiment.mkdir(exist_ok=True, parents=True)
    run = get_run_name(hyp, experiment)

    paths = {
        'experiment': experiment,
        'run': run
    }
    for name, path in paths.items():
        path.mkdir(exist_ok=True, parents=True)

    return paths


def get_run_name(hyp, experiment):
    if 'run-name' in hyp.keys():
        #  experiments/results/lunar/test
        run = experiment / hyp['run-name']

    else:
        #  experiments/results/lunar/run-0
        run = int(get_latest_run(experiment)) + 1
        run = experiment / f'run-{run}'

    return run


def make_logger(log_file, home):
    """info to STDOUT, debug to file"""
    level = logging.DEBUG

    # Create a custom logger
    fldr = home / 'logs'
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


class Writer:
    def __init__(self, name, counters, home):
        path = home / 'tensorboard' / name
        self.writer = tf.summary.create_file_writer(str(path))
        self.counters = counters

    def scalar(self, value, name, counter, verbose=False):
        value = np.array(value)

        with self.writer.as_default():
            step = self.counters[counter]
            tf.summary.scalar(name, np.mean(value), step=step)

        if verbose:
            print(f'{name} \n {counter} \n {self.counters[counter]:6.0f}, mu {np.mean(value):5.1f}, sig {np.std(value):5.1f}')
