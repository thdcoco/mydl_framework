# mydl_framework/optimizers/sgd.py

class SGD:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            # grad가 None일 수도 있으므로, 있는 경우에만 업데이트
            if hasattr(p, 'grad') and p.grad is not None:
                p.data -= self.lr * p.grad.data

    def zero_grad(self):
        for p in self.params:
            p.grad = None
