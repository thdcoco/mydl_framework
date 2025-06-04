# mydl_framework/autodiff/functions.py

import numpy as np
from .core import Function, as_array, as_variable, Variable

# -----------------------------------
# 1) square
# -----------------------------------
class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x_var, = self.inputs
        # x_var.data: 넘파이 ndarray
        return gy * 2 * x_var.data

def square(x):
    """
    x: Variable 또는 numpy scalar/ndarray
    결과로 Variable(x.data ** 2)를 반환
    """
    x = as_variable(x)
    return Square()(x)

# -----------------------------------
# 2) add
# -----------------------------------
class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        # dy/dx0 = 1, dy/dx1 = 1
        return gy, gy

def add(x0, x1):
    """
    x0, x1: Variable 또는 numpy scalar/ndarray
    결과로 Variable(x0.data + x1.data)를 반환
    """
    x0 = as_variable(x0)
    x1 = as_variable(x1)
    return Add()(x0, x1)

# -----------------------------------
# 3) mul
# -----------------------------------
class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0_var, x1_var = self.inputs
        # 순전파 때 저장된 self.inputs[0].data, self.inputs[1].data를 사용
        return gy * x1_var.data, gy * x0_var.data

def mul(x0, x1):
    """
    x0, x1: Variable 또는 numpy scalar/ndarray
    결과로 Variable(x0.data * x1.data)를 반환
    """
    x0 = as_variable(x0)
    x1 = as_variable(x1)
    return Mul()(x0, x1)

# -----------------------------------
# 4) matmul (Linear 레이어 용)
# -----------------------------------
class MatMul(Function):
    def forward(self, x, W):
        # x: 넘파이 ndarray (batch, in_features)
        # W: 넘파이 ndarray (in_features, out_features)
        return x.dot(W)

    def backward(self, gy):
        x_var, W_var = self.inputs
        x, W = x_var.data, W_var.data  # NumPy ndarray
        gx = gy.dot(W.T)
        gW = x.T.dot(gy)
        return gx, gW

def matmul(x, W):
    """
    x, W: Variable 또는 numpy 배열
    결과로 Variable(x.dot(W))를 반환
    """
    x = as_variable(x)
    W = as_variable(W)
    return MatMul()(x, W)

# -----------------------------------
# 5) ReLU
# -----------------------------------
class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x_var, = self.inputs
        x = x_var.data
        mask = x > 0
        return gy * mask

def relu(x):
    """
    x: Variable 또는 numpy 배열
    결과로 Variable(np.maximum(x.data, 0))를 반환
    """
    x = as_variable(x)
    return ReLU()(x)

# -----------------------------------
# 6) Sigmoid
# -----------------------------------
class Sigmoid(Function):
    def forward(self, x):
        y = 1.0 / (1.0 + np.exp(-x))
        self.y = y  # backward에 사용하기 위해 저장
        return y

    def backward(self, gy):
        y = self.y  # 순전파 때 계산된 값
        return gy * y * (1.0 - y)

def sigmoid(x):
    """
    x: Variable 또는 numpy 배열
    결과로 Variable(sigmoid(x.data))를 반환
    """
    x = as_variable(x)
    return Sigmoid()(x)

# -----------------------------------
# 7) Tanh
# -----------------------------------
class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        self.y = y  # backward에 사용하기 위해 저장
        return y

    def backward(self, gy):
        y = self.y
        return gy * (1.0 - y ** 2)

def tanh(x):
    """
    x: Variable 또는 numpy 배열
    결과로 Variable(np.tanh(x.data))를 반환
    """
    x = as_variable(x)
    return Tanh()(x)
