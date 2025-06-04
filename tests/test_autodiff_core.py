# tests/test_autodiff_core.py

import numpy as np
import pytest
from mydl_framework.autodiff.core import Variable, as_variable
from mydl_framework.autodiff.functions import square, add, mul

def test_variable_and_basic_ops():
    # 1) square 테스트: y = x^2 (x=3.0)
    x = Variable(np.array(3.0, dtype=np.float32))
    y = square(x)        # y = x^2 = 9.0
    assert isinstance(y, Variable)
    assert y.data == pytest.approx(9.0)

    y.backward()
    # dy/dx = 2*x = 6.0
    assert x.grad.data == pytest.approx(6.0)

    # 2) add, mul 테스트
    a = Variable(np.array(2.0, dtype=np.float32))
    b = Variable(np.array(5.0, dtype=np.float32))
    c = add(a, b)        # c = 7.0
    d = mul(a, b)        # d = 10.0
    assert c.data == pytest.approx(7.0)
    assert d.data == pytest.approx(10.0)

    d.backward()
    # ∂(a*b)/∂a = b = 5.0, ∂(a*b)/∂b = a = 2.0
    assert a.grad.data == pytest.approx(5.0)
    assert b.grad.data == pytest.approx(2.0)
