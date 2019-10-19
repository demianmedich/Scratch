# coding: utf-8

import numpy as np

from abc import ABCMeta
from abc import abstractmethod

from scratch.func import sigmoid
from scratch.func import relu
from scratch.func import softmax
from scratch.func import cross_entropy


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


class BaseLossLayer(metaclass=ABCMeta):
    def __init__(self):
        self.output = None
        self.label = None
        self.loss = None

    @abstractmethod
    def forward(self, x, t):
        pass

    @abstractmethod
    def backward(self, dout=1):
        pass


class SoftmaxWithLoss(BaseLossLayer):

    def __init__(self):
        super().__init__()

    def forward(self, x, t):
        self.output = softmax(x, axis=-1)
        self.label = t
        if t.size == self.output.size:
            # t가 one-hot vector 인 경우에 label 값으로 변경해준다.
            # 변경되면 1d array로 바뀌며, 이렇게 하는 이유는 backprop 에서 계산량을
            # 줄이기 위해서이다.
            self.label = self.output.argmax(axis=-1)
        self.loss = cross_entropy(self.output, self.label)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.label.shape[0]
        # cross entropy 와 같이 미분하면 y1 - t1, y2 - t2 같은 식으로 기울기가 구해진다
        # t는 one-hot vector 라서 실제 정답 레이블인 경우만 계산한다.
        dx = self.output.copy()
        dx[np.arange(batch_size), self.label] -= 1
        dx *= dout
        dx /= batch_size  # batch 사이즈로 나누는 이유는..??
        return dx
