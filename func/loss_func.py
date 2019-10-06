# coding: utf-8

import numpy as np


def cross_entropy(y_pred, y_true):
    assert isinstance(y_pred, np.ndarray), "y_pred should be one of the instance of numpy.ndarray."
    assert isinstance(y_true, np.ndarray), "y_true should be one of the instance of numpy.ndarray."

    assert y_pred.ndim == y_true.ndim, "y_pred and y_true should have same ndim value."

    assert y_pred.ndim == 1 or y_pred.ndim == 2, \
        "ndim value of y_pred should be 1 or 2 but {}.".format(y_pred.ndim)

    if y_pred.ndim == 1:
        y_pred = np.reshape(y_pred, [1, y_pred.size])
        y_true = np.reshape(y_true, [1, y_true.size])

    assert y_pred.shape[0] == y_true.shape[0], \
        "batch size(0 dim) of y_pred, y_true should be equal."

    if y_true.size == y_pred.size:
        # keepdims=False so return value will be 1d array
        y_true = y_true.argmax(axis=1)

    batch_size = y_pred.shape[0]
    return -np.sum(np.log(y_pred[np.arange(batch_size), y_true] + 1e-7)) / batch_size
