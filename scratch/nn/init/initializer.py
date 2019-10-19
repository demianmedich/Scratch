# coding: utf-8

import numpy as np

from abc import ABCMeta
from abc import abstractmethod


class Initializer(metaclass=ABCMeta):

    @abstractmethod
    def generate(self):
        pass

    # def __call__(self, *args, **kwargs):
    #     self.generate()


class Ones(Initializer):
    __slots__ = ['shape', 'dtype']

    def __init__(self, shape, dtype=np.float32):
        super().__init__()
        self.shape = shape
        self.dtype = dtype

    def generate(self):
        return np.ones(self.shape, self.dtype)


class Zeros(Initializer):
    __slots__ = ['shape', 'dtype']

    def __init__(self, shape, dtype=np.float32):
        super().__init__()
        self.shape = shape
        self.dtype = dtype

    def generate(self):
        return np.zeros(self.shape, self.dtype)


class Normal(Initializer):
    __slots__ = ['shape', 'mean', 'variance']

    def __init__(self, shape, mean=0, variance=1):
        super().__init__()
        self.shape = shape
        self.mean = mean
        self.variance = variance

    def generate(self):
        return np.random.normal(size=self.shape, loc=self.mean, scale=self.variance)


class StandardNormal(Normal):

    def __init__(self, shape):
        super().__init__(shape, mean=0, variance=1)
