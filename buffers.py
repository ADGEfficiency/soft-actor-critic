import pickle

import numpy as np

import utils


def make(
    env,
    hyp
):
    buffer_path = hyp.get('buffer')

    if buffer_path == 'new':
        return Buffer(env.elements, size=hyp['buffer-size'])

    elif buffer_path is None:
        return Buffer(env.elements, size=hyp['buffer-size'])

    elif buffer_path == 'lastest':
        last_run = utils.get_latest_run()
        buffer = load_buffer(last_run)
        assert buffer.full
        return buffer

    else:
        buffer = load_buffer(buffer_path)
        assert buffer.full
        return buffer


def save(buffer, path, name):
    path = path / 'buffers' / name
    print(f'saving buffer to {path}')
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('wb') as fi:
        pickle.dump(buffer, fi)


def load_buffer():
    print(f'loading buffer from {bpath}')
    with bpath.open('rb') as fi:
        return pickle.load(fi)


class Buffer():
    def __init__(self, elements, size=64):
        self.data = {
            el: np.zeros((size, *shape), dtype=dtype) for el, shape, dtype in elements
        }
        self.size = size
        self.cursor = 0
        self.full = False

    def __len__(self):
        return len(self.data['observation'])

    @property
    def cursor(self):
        return self._cursor

    @cursor.setter
    def cursor(self, value):
        if value == self.size:
            self._cursor = 0
            self.full = True
        else:
            self._cursor = value

    def append(self, data):
        for name, el in zip(data._fields, data):
            self.data[name][self.cursor, :] = el
        self.cursor = self.cursor + 1

    def sample(self, num):
        if not self.full:
            raise ValueError("buffer is not full!")
        idx = np.random.randint(0, self.size, num)
        batch = {}
        for name, data in self.data.items():
            batch[name] = data[idx, :]
        return batch
