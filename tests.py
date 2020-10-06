from collections import namedtuple

import numpy as np

from buffer import Buffer


def test_buffer():
    elements = (
        ('A', (1,), 'int32'),
        ('B', (2,), 'float32'),
    )

    tup = namedtuple('tup', ('A', 'B'))

    buff = Buffer(elements, size=4)

    data = ((0, (1, 1)), (1, (2, 2)), (2, (3, 3)))
    data = [tup(*d) for d in data]

    for _ in range(2):
        for d in data:
            buff.append(d)

    batch = buff.sample(2)

    for a, b in zip(batch['A'], batch['B']):
        check = data[int(a)]
        np.testing.assert_array_equal(b, check[1])

    np.testing.assert_array_equal(buff.data['A'], np.array([1, 2, 2, 0]).reshape(4, 1))
    np.testing.assert_array_equal(buff.data['B'], np.array(((2, 2), (3, 3), (3, 3), (1, 1))).reshape(4, 2))
