# examples/llm_mnist_debug.py

import os
import json
import matplotlib.pyplot as plt

from mydl_framework.llm_support.gpt_client import GPTClient
from mydl_framework.llm_support.model_builder import ModelBuilder
from mydl_framework.datasets.mnist_loader import MNISTLoader
from mydl_framework.optimizers.sgd import SGD
from mydl_framework.training.trainer import Trainer

# 0. (선택) 환경변수로 세팅되어 있지 않다면 직접
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# 1. LLM → JSON 스펙
nl_desc = "3-layer MLP, hidden 256, ReLU, output 10, CrossEntropy"
client = GPTClient()  
spec = client.to_graph_input(nl_desc)
print(" LLM이 생성한 스펙:")
print(json.dumps(spec, indent=2))

# 2. ModelBuilder → 모델 생성
model = ModelBuilder.from_spec(spec, input_dim=784)
print("\n 생성된 모델 레이어 리스트:")
for i, layer in enumerate(model.layers):
    params = getattr(layer, "params", None)
    print(f"  Layer {i}: {layer.__class__.__name__}, params count = {len(params) if params else 0}")

# 3. DataLoader & Trainer 준비
train_loader = MNISTLoader(train=True, batch_size=64)
val_loader   = MNISTLoader(train=False, batch_size=100)
optimizer    = SGD(model.parameters(), lr=0.01)
trainer      = Trainer(model, optimizer, train_loader, val_loader, num_epochs=5)

# 4. 학습 중 loss/accuracy를 리스트로 수집할 수 있게 함수 래핑
loss_history = []
acc_history  = []

original_train = trainer.train_one_epoch
original_val   = trainer.validate_one_epoch

def train_and_record(epoch):
    original_train(epoch)
    # 마지막 printed loss를 parser 대신, Trainer가 내부적으로 저장하도록 개선하면 더 좋습니다
    # 여기서는 그냥 console 출력을 loss_history에 append한다고 가정
    # 실제로는 Trainer에 return loss value 기능을 추가하세요

def val_and_record(epoch):
    original_val(epoch)
    # 마찬가지로 accuracy를 acc_history에 append

trainer.train_one_epoch    = train_and_record
trainer.validate_one_epoch = val_and_record

# 5. 학습 실행
trainer.fit()

# 6. 학습 곡선 그리기
plt.figure()
plt.plot(loss_history, label="Train Loss")
plt.plot(acc_history, label="Val Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.title("Training Curve")
plt.show()
