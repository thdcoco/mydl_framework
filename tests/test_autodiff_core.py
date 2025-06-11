import pytest
from testrix.autodiff.variable import Variable
from testrix.autodiff.function import add, mul

def test_add_mul():
    x = Variable(2.0)
    y = Variable(3.0)
    z = add(x, y)
    z.backward()
    assert x.grad.data == 1.0
    assert y.grad.data == 1.0
    u = mul(x, y)
    u.backward()
    assert x.grad.data == y.data