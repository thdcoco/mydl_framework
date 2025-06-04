# mydl_framework/training/trainer.py

import numpy as np
from mydl_framework.autodiff.core import Variable

class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader=None, epochs=10, device=None):
        """
        model: mydl_framework.layers.Model 을 상속받은 객체
        optimizer: mydl_framework.optimizers.SGD 등
        loss_fn: mydl_framework.layers.losses.CrossEntropyLoss
        train_loader: mydl_framework.datasets.DataLoader 객체 (학습용)
        val_loader: mydl_framework.datasets.DataLoader 객체 (검증용)
        epochs: 학습 반복 횟수
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            # --- 학습 단계 ---
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for batch_x, batch_y in self.train_loader:
                # batch_x: (batch_size, 28,28) 이미지 데이터
                # batch_y: (batch_size,) 레이블

                # 1) (batch_size, 784) 형태로 변환
                x = batch_x.reshape(batch_x.shape[0], -1)
                x_var = Variable(x)

                # 2) 순전파
                logits = self.model(x_var)  # (batch, num_classes)

                # 3) 손실 계산
                loss = self.loss_fn(logits, batch_y)

                # 4) 역전파 → 업데이트
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 5) 통계 집계
                total_loss += float(loss.data) * batch_x.shape[0]
                preds = np.argmax(logits.data, axis=1)
                total_correct += np.sum(preds == batch_y)
                total_samples += batch_x.shape[0]

            train_loss = total_loss / total_samples
            train_acc = total_correct / total_samples
            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

            # --- 검증 단계 (선택) ---
            if self.val_loader is not None:
                val_loss = 0.0
                val_correct = 0
                val_samples = 0
                for batch_x, batch_y in self.val_loader:
                    x = batch_x.reshape(batch_x.shape[0], -1)
                    x_var = Variable(x)
                    logits = self.model(x_var)

                    loss = self.loss_fn(logits, batch_y)
                    val_loss += float(loss.data) * batch_x.shape[0]
                    preds = np.argmax(logits.data, axis=1)
                    val_correct += np.sum(preds == batch_y)
                    val_samples += batch_x.shape[0]

                val_loss /= val_samples
                val_acc = val_correct / val_samples
                print(f"         | Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
