# examples/llm_mnist_debug_tune.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import confusion_matrix, classification_report
    has_sklearn = True
except ImportError:
    print("⚠️ scikit-learn이 없어 혼동 행렬/분류 리포트를 건너뜁니다.")
    has_sklearn = False

from mydl_framework.llm_support.gpt_client import GPTClient
from mydl_framework.llm_support.model_builder import ModelBuilder
from mydl_framework.datasets.mnist_loader import MNISTLoader
from mydl_framework.optimizers.adam import Adam
from mydl_framework.training.trainer import Trainer

# --- Configuration ---
NL_DESC        = "3-layer MLP, hidden 256, ReLU, output 10, CrossEntropy"
INPUT_DIM      = 784
BATCH_SIZE     = 64
VAL_BATCH_SIZE = 100
LR             = 1e-3
EPOCHS         = 30
# ----------------------

# 0. Ensure API key is set
os.environ.setdefault("OPENAI_API_KEY", "your_openai_api_key_here")

# 1. Generate model spec via LLM
client   = GPTClient()
raw_spec = client.to_graph_input(NL_DESC)
print("🔍 Final spec:", json.dumps(raw_spec, indent=2))

# 2. Build model from spec
model = ModelBuilder.from_spec(raw_spec, input_dim=INPUT_DIM)
print("📐 Layers:")
for i, layer in enumerate(model.layers):
    params = getattr(layer, 'params', None)
    print(f"  Layer {i}: {layer.__class__.__name__}, params={len(params) if params else 0}")

# 3. Prepare data loaders
train_loader = MNISTLoader(train=True,  batch_size=BATCH_SIZE)
val_loader   = MNISTLoader(train=False, batch_size=VAL_BATCH_SIZE)

# 4. Create optimizer and trainer
optimizer = Adam(model.parameters(), lr=LR)
trainer   = Trainer(model, optimizer, train_loader, val_loader, num_epochs=EPOCHS)

# 5. Train and visualize learning curves
trainer.fit()

# 6. Evaluate with confusion matrix & classification report
all_preds, all_labels = [], []
for x, y in val_loader:
    out = trainer.model(x)
    preds = out.data.argmax(axis=1)
    all_preds.extend(preds.tolist())
    all_labels.extend(y.tolist())

if has_sklearn:
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(all_labels, all_preds, digits=4))
else:
    print("\n혼동 행렬 및 분류 리포트 생략됨.")
