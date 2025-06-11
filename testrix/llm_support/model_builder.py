import numpy as np
from testrix.autodiff.variable import Variable
from testrix.layers.linear import Linear
from testrix.layers.activations import ReLU, Sigmoid, Tanh
from testrix.layers.softmax_cross_entropy import SoftmaxCrossEntropy

class ModelBuilder:
    @staticmethod
    def from_spec(spec: dict, input_dim: int):
        layers = []
        current_dim = input_dim

        for cfg in spec.get("layers", []):
            t = cfg["type"].lower()
            if t == "linear":
                out_dim = cfg["out_features"]
                layers.append(Linear(current_dim, out_dim))
                current_dim = out_dim
            elif t == "relu":
                layers.append(ReLU())
            elif t == "sigmoid":
                layers.append(Sigmoid())
            elif t == "tanh":
                layers.append(Tanh())
            else:
                # 이미 gpt_client에서 필터링했으니 여기서는 안전
                continue

        # 손실 함수는 Trainer에서 SoftmaxCrossEntropy로 처리
        class SimpleModel:
            def __init__(self, layers):
                self.layers = layers

            def __call__(self, x):
                if isinstance(x, np.ndarray):
                    x = Variable(x)
                out = x
                for layer in self.layers:
                    out = layer(out)
                return out

            def parameters(self):
                ps = []
                for l in self.layers:
                    if hasattr(l, "params"):
                        ps.extend(l.params)
                return ps

        return SimpleModel(layers)
