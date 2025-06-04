# mydl_framework/layers/activations.py

from mydl_framework.autodiff.functions import relu, sigmoid, tanh

class ReLU:
    def __call__(self, x):
        # x: Variable
        return relu(x)

class Sigmoid:
    def __call__(self, x):
        return sigmoid(x)

class Tanh:
    def __call__(self, x):
        return tanh(x)
