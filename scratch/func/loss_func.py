# coding: utf-8

import numpy as np


def cross_entropy(y_pred, y_true):
    assert isinstance(y_pred, np.ndarray), \
        "y_pred should be one of the instance of numpy.ndarray."
    assert isinstance(y_true, np.ndarray), \
        "y_true should be one of the instance of numpy.ndarray."

    assert y_pred.ndim == 2, \
        "ndim value of y_pred should 2 but {}.".format(y_pred.ndim)

    assert y_pred.shape[0] == y_true.shape[0], \
        "batch size(0 dim) of y_pred, y_true should be equal."

    if y_true.size == y_pred.size:
        # label 이 one-hot vector 라면 인덱스 값들로 변경되도록 고친다.
        y_true = y_true.argmax(axis=1)

    batch_size = y_pred.shape[0]
    # 크로스 엔트로피의 정의는 -sum(t * log(y)) 이다.
    # 여기서 t가 one-hot vector 이기 때문에,
    # 실제로 정답인 경우 (t가 1인 경우)의 log(y) 값이 사용되면 되기 때문에,
    # sum 값도 사실은 정답 레이블에 매칭되는 log(y) 값만 알면 된다.
    # 따라서 불필요하게 matrix 계산을 시키지 않고, log(y) 값만 가져온다.
    # 그리고 mini-batch 전체에 대해 평균을 취한다.
    return -np.sum(
        np.log(y_pred[np.arange(batch_size), y_true] + 1e-7)) / batch_size
