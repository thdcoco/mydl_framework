import numpy as np
from mydl_framework.autodiff.variable import Variable
from mydl_framework.autodiff.function import Function

class Linear(Function):
    def __init__(self, in_features, out_features):
        self.W = Variable(np.random.randn(in_features, out_features) * np.sqrt(2/in_features))
        self.b = Variable(np.zeros(out_features))
        self.params = [self.W, self.b]

    def forward(self, x):
        self.x = x
        return x.dot(self.W.data) + self.b.data

    def backward(self, gy):
        gx = gy.dot(self.W.data.T)
        self.W.grad = Variable(self.x.T.dot(gy))
        self.b.grad = Variable(np.sum(gy, axis=0))
        return gx
