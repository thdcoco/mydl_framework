# mydl_framework/training/trainer.py

import matplotlib.pyplot as plt
from mydl_framework.autodiff.variable import Variable
from mydl_framework.layers.softmax_cross_entropy import SoftmaxCrossEntropy

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, num_epochs=10):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        # Softmax + CrossEntropy 결합된 Function
        self.loss_fn = SoftmaxCrossEntropy()

    def fit(self):
        loss_history, acc_history = [], []
        for epoch in range(self.num_epochs):
            loss = self.train_one_epoch(epoch)
            acc  = self.validate_one_epoch(epoch)
            loss_history.append(loss)
            acc_history.append(acc)
        # 학습 곡선 시각화
        plt.figure()
        plt.plot(loss_history, label="훈련 손실")
        plt.plot(acc_history,  label="검증 정확도")
        plt.xlabel("에포크")
        plt.ylabel("값")
        plt.legend()
        plt.title("학습 곡선")
        plt.show()

    def train_one_epoch(self, epoch):
        total_loss = 0.0
        batches = 0
        for x, y in self.train_loader:
            # 순전파: 모델이 NumPy 배열을 Variable로 변환
            out_var = self.model(x)
            # 레이블도 Variable로 래핑
            t_var = Variable(y)
            # 손실 계산 (Function 호출)
            loss_var = self.loss_fn(out_var, t_var)
            # 역전파
            loss_var.backward()
            # 파라미터 업데이트
            self.optimizer.step()
            self.optimizer.zero_grad()
            # 누적 손실 및 배치 카운트
            total_loss += float(loss_var.data)
            batches += 1

        avg_loss = total_loss / batches
        print(f"Epoch {epoch} training loss: {avg_loss:.4f}")
        return avg_loss

    def validate_one_epoch(self, epoch):
        correct, total = 0, 0
        for x, y in self.val_loader:
            out_var = self.model(x)
            preds = out_var.data.argmax(axis=1)
            correct += (preds == y).sum()
            total   += len(y)
        acc = correct / total
        print(f"Epoch {epoch} validation accuracy: {acc:.4f}")
        return acc
