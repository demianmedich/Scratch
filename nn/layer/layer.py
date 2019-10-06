# coding: utf-8

import numpy as np

from abc import ABCMeta
from abc import abstractmethod

from func import sigmoid
from func import relu
from func import softmax
from func import cross_entropy
from nn.init import Normal
from nn.init import Zeros


class BaseLayer(metaclass=ABCMeta):

    def __init__(self):
        self.params = None
        self.grads = None
        self.output = None

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dout):
        pass


class Sigmoid(BaseLayer):

    def forward(self, x):
        self.output = sigmoid(x)
        return self.output

    def backward(self, dout):
        dx = dout * self.output * (1.0 - self.output)
        return dx


class ReLU(BaseLayer):

    def forward(self, x):
        self.output = relu(x)
        return self.output

    def backward(self, dout):
        dx = dout * (1.0 if self.output > 0.0 else 0.0)
        return


class MatMul(BaseLayer):

    def __init__(self, weights):
        super().__init__()

        self.params = [weights]
        self.grads = [np.zeros_like(weights)]
        self.x = None

    def forward(self, x):
        W = self.params[0]
        out = np.matmul(x, W)
        self.x = x

        return out

    def backward(self, dout):
        W, b = self.params[0], self.params[1]
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)

        self.grads[0][...] = dW

        return dx


class Affine(BaseLayer):

    def __init__(self, weights, bias):
        super().__init__()

        self.params = [weights, bias]
        self.grads = [np.zeros_like(weights), np.zeros_like(bias)]
        self.x = None

    def forward(self, x):
        W, b = self.params[0], self.params[1]
        self.output = np.matmul(x, W) + b
        self.x = x
        return self.output

    def backward(self, dout):
        W, b = self.params[0], self.params[1]
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class Softmax(BaseLayer):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.output = softmax(x, axis=-1)
        return self.output

    def backward(self, dout):
        dx = self.output * dout
        sum_dx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.output * sum_dx
        return dx
