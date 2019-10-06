# coding: utf-8

import numpy as np

from nn.layer import Affine, Sigmoid


class TwoLayerNet:

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size):
        W1 = np.random.randn(input_size, hidden_size)
        b1 = np.zeros(hidden_size)
        W2 = np.random.randn(hidden_size, output_size)
        b2 = np.zeros(output_size)

        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        self.params = [layer.params for layer in self.layers]

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
