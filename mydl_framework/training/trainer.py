import matplotlib.pyplot as plt
import numpy as np
from mydl_framework.autodiff.variable import Variable
from mydl_framework.layers.softmax_cross_entropy import SoftmaxCrossEntropy

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, num_epochs=5):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.loss_fn = SoftmaxCrossEntropy()

    def fit(self):
        loss_history = []
        acc_history = []
        for epoch in range(self.num_epochs):
            loss = self.train_one_epoch(epoch)
            acc = self.validate_one_epoch(epoch)
            loss_history.append(loss)
            acc_history.append(acc)
        plt.figure()
        plt.plot(loss_history, label='Train Loss')
        plt.plot(acc_history, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.title('Training Curve')
        plt.show()

    def train_one_epoch(self, epoch):
        total_loss = 0
        batches = 0
        for x, y in self.train_loader:
            out_var = self.model(x)
            logits = out_var.data if isinstance(out_var, Variable) else out_var
            loss = self.loss_fn(logits, y)
            loss_var = Variable(loss)
            grad = self.loss_fn.backward()
            loss_var.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss
            batches += 1
        avg_loss = total_loss / batches
        print(f"Epoch {epoch} training loss: {avg_loss:.4f}")
        return avg_loss

    def validate_one_epoch(self, epoch):
        correct = 0
        total = 0
        for x, y in self.val_loader:
            out_var = self.model(x)
            preds = out_var.data.argmax(axis=1)
            correct += (preds == y).sum()
            total += len(y)
        acc = correct / total
        print(f"Epoch {epoch} validation accuracy: {acc:.4f}")
        return acc