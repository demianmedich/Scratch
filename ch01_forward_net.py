# coding: utf-8
import numpy as np
from scratch.nn.simple_nn import TwoLayerNet
from scratch.func import cross_entropy


if __name__ == '__main__':
    x = np.random.randn(10, 2)
    model = TwoLayerNet(2, 4, 3)
    predict = model.predict(x)
    print(x)
    print(predict)

    labels = np.zeros([10, 3])
    loss = cross_entropy(predict, labels)
    print(loss)
