# mydl_framework/autodiff/function.py

import numpy as np

class Function:
    def __init__(self):
        # 자동미분 그래프 세대를 관리하기 위한 초기값
        self.generation = 0

    def __call__(self, *inputs):
        # 호출 직전에, 입력 Variable들의 세대 중 최대값을 자신의 세대로 설정
        self.generation = max(x.generation for x in inputs)
        # Variable은 여기서만 임포트하여 순환참조 방지
        from testrix.autodiff.variable import Variable

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(y) for y in ys]
        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs
        return outputs[0] if len(outputs) == 1 else outputs

    def forward(self, *xs):
        raise NotImplementedError

    def backward(self, *gys):
        raise NotImplementedError


class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1.data, gy * x0.data

def mul(x0, x1):
    return Mul()(x0, x1)
