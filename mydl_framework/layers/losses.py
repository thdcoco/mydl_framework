# mydl_framework/layers/activations.py

import numpy as np
from mydl_framework.autodiff.core import Variable
from mydl_framework.autodiff.functions import relu, sigmoid, tanh

class ReLU:
    def forward(self, x: Variable) -> Variable:
        return relu(x)

    def parameters(self):
        return []

class Sigmoid:
    def forward(self, x: Variable) -> Variable:
        return sigmoid(x)

    def parameters(self):
        return []

class Tanh:
    def forward(self, x: Variable) -> Variable:
        return tanh(x)

    def parameters(self):
        return []
