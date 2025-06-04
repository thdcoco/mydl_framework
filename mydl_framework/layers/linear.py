# mydl_framework/layers/linear.py

import numpy as np
from mydl_framework.autodiff.core import Variable
from mydl_framework.autodiff.functions import matmul  # 반드시 포함

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        W_data = np.random.randn(in_features, out_features).astype(np.float32)
        self.W = Variable(W_data)
        if bias:
            b_data = np.zeros(out_features, dtype=np.float32)
            self.b = Variable(b_data)
        else:
            self.b = None

    def __call__(self, x):
        y = matmul(x, self.W)
        if self.b is not None:
            y = y + self.b
        return y
