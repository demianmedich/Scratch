# coding: utf-8

from abc import ABCMeta
from abc import abstractmethod


class BaseOptimizer(metaclass=ABCMeta):

    @abstractmethod
    def update(self, params, grads):
        pass


class SGD(BaseOptimizer):

    def __init__(self, learning_rate):
        self.learning_late = learning_rate

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.learning_late * grads[i]
