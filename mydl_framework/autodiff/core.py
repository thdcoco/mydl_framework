# mydl_framework/autodiff/core.py

import weakref
import numpy as np
import contextlib

# ------------------------------------------------------------------------------
# Config: 역전파 활성화/비활성화 조절
# ------------------------------------------------------------------------------
class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

# ------------------------------------------------------------------------------
# Variable: 기본 텐서 래퍼 클래스
# ------------------------------------------------------------------------------
class Variable:
    __array_priority__ = 200  # NumPy 연산자 오버라이딩 우선순위

    def __init__(self, data, name=None):
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError(f"{type(data)} is not supported. Use numpy.ndarray.")
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return f"variable({p})"

    # ------------- 연산자 오버로드 -------------
    def __add__(self, other):
        from mydl_framework.autodiff.functions import add
        if isinstance(other, Variable):
            return add(self, other)
        return add(self, Variable(np.array(other)))

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        from mydl_framework.autodiff.functions import mul
        if isinstance(other, Variable):
            return mul(self, other)
        return mul(self, Variable(np.array(other)))

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        from mydl_framework.autodiff.functions import neg
        return neg(self)

    def __sub__(self, other):
        from mydl_framework.autodiff.functions import sub
        if isinstance(other, Variable):
            return sub(self, other)
        return sub(self, Variable(np.array(other)))

    def __rsub__(self, other):
        from mydl_framework.autodiff.functions import rsub
        return rsub(self, other)

    def __truediv__(self, other):
        from mydl_framework.autodiff.functions import div
        if isinstance(other, Variable):
            return div(self, other)
        return div(self, Variable(np.array(other)))

    def __rtruediv__(self, other):
        from mydl_framework.autodiff.functions import rdiv
        return rdiv(self, other)

    def __pow__(self, power):
        from mydl_framework.autodiff.functions import pow
        return pow(self, power)

    # ------------------------------------------

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set and f is not None:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]  # weakref → Variable
            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx
                    if x.creator is not None:
                        add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # weakref → Variable

# ------------------------------------------------------------------------------
# Helper: Python 객체 → Variable / NumPy 배열 변환
# ------------------------------------------------------------------------------
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

# ------------------------------------------------------------------------------
# Function: 모든 연산 함수의 부모 클래스
# ------------------------------------------------------------------------------
class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # 순전파(NumPy ndarray)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()
