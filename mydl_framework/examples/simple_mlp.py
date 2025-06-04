# mydl_framework/examples/simple_mlp.py

import os
import numpy as np

from mydl_framework.layers import Linear, ReLU, Softmax, CrossEntropyLoss, Model
from mydl_framework.datasets.mnist import MNIST
from mydl_framework.optimizers.sgd import SGD
from mydl_framework.training.trainer import Trainer

class SimpleMLP(Model):
    def __init__(self, input_size=784, hidden1=256, hidden2=256, output_size=10):
        super().__init__()
        self.l1 = Linear(input_size, hidden1)
        self.act1 = ReLU()
        self.l2 = Linear(hidden1, hidden2)
        self.act2 = ReLU()
        self.l3 = Linear(hidden2, output_size)
        self.loss = CrossEntropyLoss()
        # 모델 레이어를 순서대로 리스트에 저장
        self.layers = [self.l1, self.act1, self.l2, self.act2, self.l3]

    def forward(self, x):
        # x: (batch, 784) 모양의 Variable
        h = self.l1(x)
        h = self.act1(h)
        h = self.l2(h)
        h = self.act2(h)
        logits = self.l3(h)
        return logits

    def parameters(self):
        # Model 부모 클래스의 parameters()를 쓰려면 layers 속성 필요
        return super().parameters()

def load_mnist(batch_size=64):
    """MNIST dataset을 로드하여 DataLoader로 감싸 줍니다."""
    train_ds = MNIST(root=os.path.join(os.getcwd(), "data"), train=True)
    test_ds = MNIST(root=os.path.join(os.getcwd(), "data"), train=False)

    # DataLoader: 배치별로 반복 가능한 객체 (가정: DataLoader가 구현되어 있다고 할 때)
    from mydl_framework.datasets.base import DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

if __name__ == "__main__":
    # 1) Hyperparameter 세팅
    epochs = 5
    learning_rate = 0.1
    batch_size = 64

    # 2) 데이터 로드
    print("▶ Loading MNIST...")
    train_loader, val_loader = load_mnist(batch_size=batch_size)
    print("   Train batches:", len(train_loader), " Validation batches:", len(val_loader))

    # 3) 모델 생성
    model = SimpleMLP()
    print("▶ Model created:", model)

    # 4) Optimizer (SGD) 생성
    optimizer = SGD(model.parameters(), lr=learning_rate)

    # 5) Trainer 초기화 후 학습 시작
    print("▶ Start training SimpleMLP on MNIST!")
    trainer = Trainer(model, optimizer, loss_fn=model.loss)
    trainer.fit(train_loader, val_loader, epochs=epochs)

    print("▶ Training finished!")
