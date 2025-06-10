# mydl_framework/llm_support/model_builder.py
import numpy as np
from mydl_framework.layers import Linear, ReLU, Sigmoid, Tanh, CrossEntropy
from mydl_framework.autodiff.variable import Variable

class ModelBuilder:
    @staticmethod
    def from_spec(spec: dict, input_dim: int):
        layers = []
        current_dim = input_dim
        for layer_cfg in spec.get("layers", []):
            t = layer_cfg.get("type")
            if t == "Linear":
                out_f = layer_cfg.get("out_features")
                layers.append(Linear(current_dim, out_f))
                current_dim = out_f
            elif t in ("ReLU", "Sigmoid", "Tanh"):  # activation
                if t == "ReLU":
                    layers.append(ReLU())
                elif t == "Sigmoid":
                    layers.append(Sigmoid())
                else:
                    layers.append(Tanh())
            else:
                raise ValueError(f"Unsupported layer type: {t}")
        # 손실 함수
        loss_name = spec.get("loss", "CrossEntropy")
        if loss_name == "CrossEntropy":
            loss = CrossEntropy()
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

        class SimpleModel:
            def __init__(self, layers, loss_fn):
                self.layers = layers
                self.loss_fn = loss_fn

            def __call__(self, x):
                # NumPy 배열이면 Variable로 자동 래핑
                if isinstance(x, np.ndarray):
                    x = Variable(x)
                out = x
                for layer in self.layers:
                    out = layer(out)
                return out

            def parameters(self):
                params = []
                for layer in self.layers:
                    if hasattr(layer, 'params'):
                        params.extend(layer.params)
                return params

        return SimpleModel(layers, loss)
