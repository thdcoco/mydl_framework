# mydl_framework/autodiff/variable.py

import numpy as np

class Variable:
    def __init__(self, data, name=None):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):
        # 1) gradient 초기화
        self.grad = Variable(np.ones_like(self.data))
        # 2) 역전파 함수 스택 초기화
        funcs = []
        if self.creator:
            funcs.append(self.creator)
        # 3) 역전파 수행
        while funcs:
            f = funcs.pop()
            if f is None:
                continue
            # 각 출력 Variable의 grad.data 수집
            gys = [output.grad.data for output in f.outputs if output is not None]
            # backward 계산 결과
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            # 입력 Variable에 grad 할당 및 스택 추가
            for x, gx in zip(f.inputs, gxs):
                x.grad = Variable(gx)
                if x.creator:
                    funcs.append(x.creator)

    def __add__(self, other):
        from testrix.autodiff.function import add
        return add(self, other)

    def __mul__(self, other):
        from testrix.autodiff.function import mul
        return mul(self, other)
