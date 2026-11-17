import functools

import numpy


class HashableArray:
    def __init__(self, array):
        self.array = array
        array = numpy.asarray(array)
        self.key = (array.shape, array.dtype.str, array.tobytes())

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.key == other.key


def read_only(result):
    for value in result if isinstance(result, tuple) else [result]:
        if isinstance(value, numpy.ndarray):
            value.flags.writeable = False
    return result


def array_cache(function):
    @functools.lru_cache(maxsize=1024)
    def cached(self, *args, **kwargs):
        return read_only(function(self, *[arg.array for arg in args],
                                  **{name: arg.array for name, arg in kwargs.items()}))

    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        return cached(self, *[HashableArray(arg) for arg in args],
                      **{name: HashableArray(arg) for name, arg in kwargs.items()})

    return wrapper


def composition_cache(function):
    @functools.lru_cache(maxsize=16384)
    def cached(self, x_A, x_B):
        return function(self, numpy.array([[x_A], [x_B]]))

    @functools.wraps(function)
    def wrapper(self, x):
        x = numpy.asarray(x, dtype=float)
        columns = [cached(self, x_A, x_B) for x_A, x_B in x.reshape(2, -1).T.tolist()]
        return tuple(numpy.concatenate(values).reshape(x[0].shape)
                     for values in zip(*columns))

    return wrapper
