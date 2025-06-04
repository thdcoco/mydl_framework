# mydl_framework/optimizers/sgd.py

class SGD:
    def __init__(self, params, lr=0.01):
        self.params = list(params)  # Variable 객체들의 리스트
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad.data

    def zero_grad(self):
        for p in self.params:
            p.cleargrad()
class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0):
        self.params = list(params)  # Variable 객체들의 리스트
        self.lr = lr
        self.momentum = momentum
        if momentum > 0:
            # velocity를 0으로 초기화
            self.velocities = {id(p): np.zeros_like(p.data) for p in self.params}
        else:
            self.velocities = None

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.data
            if self.momentum > 0:
                v = self.velocities[id(p)]
                v = self.momentum * v + (1 - self.momentum) * grad
                self.velocities[id(p)] = v
                p.data -= self.lr * v
            else:
                p.data -= self.lr * grad

    def zero_grad(self):
        for p in self.params:
            p.cleargrad()

class StepLR:
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0

    def step(self):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
