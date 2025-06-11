import numpy as np
from testrix.autodiff.function import Function

class ReLU(Function):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, gy):
        return gy * self.mask

class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y

    def backward(self, gy):
        return gy * (1 - self.y) * self.y

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        self.y = y
        return y

    def backward(self, gy):
        return gy * (1 - self.y ** 2)