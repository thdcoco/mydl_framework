# mydl_framework/layers/softmax_cross_entropy.py

import numpy as np
from testrix.autodiff.function import Function
from testrix.autodiff.variable import Variable

class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        # x: logits (batch, classes), t: labels (batch,)
        # 숫자 안정화
        x = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(x)
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)
        self.labels = t
        N = x.shape[0]
        loss = -np.log(self.probs[np.arange(N), t]).mean()
        return loss

    def backward(self, gy):
        # gy: upstream gradient (scalar)
        N = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.labels] -= 1
        grad = grad * (gy / N)
        return grad, None   # second None for t
