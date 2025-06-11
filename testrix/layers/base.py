import numpy as np
from testrix.autodiff.variable import Variable

class CrossEntropy:
    def __init__(self): pass
    def __call__(self, logits, labels):
        # logits shape: (batch, classes)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        N = logits.shape[0]
        loss = -np.log(probs[np.arange(N), labels]).mean()
        self.probs = probs
        self.labels = labels
        return loss

    def backward(self):
        N = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(N), self.labels] -= 1
        return grad / N
