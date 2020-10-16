import json

import tensorflow as tf


def dump_json(data, file):
    with open(file, 'w') as fi:
        json.dump(data, fi)


def minimum_target(state, action, targets):
    return tf.reduce_min([t([state, action]) for t in targets], axis=0)
