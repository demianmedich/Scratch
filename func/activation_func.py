# coding: utf-8

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return x if x > 0 else 0


def softmax(x, axis=-1):
    assert isinstance(x,
                      np.ndarray), "x should be one of the instance of numpy.ndarray"
    assert x.ndim == 2, "ndim value of x should be 2 but {}".format(x.ndim)

    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    summation = np.sum(exp, axis=axis, keepdims=True)
    return exp / summation
