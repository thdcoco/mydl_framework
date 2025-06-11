# app.py
import streamlit as st
import numpy as np
import json
from mydl_framework.llm_support.gpt_client import GPTClient
from mydl_framework.llm_support.model_builder import ModelBuilder
from mydl_framework.datasets.mnist_loader import MNISTLoader
from mydl_framework.optimizers.adam import Adam
from mydl_framework.training.trainer import Trainer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

st.title("나만의 딥러닝 프레임워크 인터랙티브 데모")

# 사이드바: 하이퍼파라미터 설정
st.sidebar.header("하이퍼파라미터 설정")
lr = st.sidebar.slider("학습률(Learning Rate)", 1e-4, 1e-1, 1e-3, format="%.4f")
batch_size = st.sidebar.selectbox("배치 크기(Batch Size)", [32, 64, 128], index=1)
epochs = st.sidebar.slider("에포크 수(Epochs)", 5, 50, 20)
hidden_units = st.sidebar.slider("은닉 유닛 수(Hidden Units)", 64, 512, 256, step=64)

st.sidebar.header("모델 스펙 입력")
nl_desc = st.sidebar.text_input("자연어로 모델 정의", 
    f"3-layer MLP, hidden {hidden_units}, ReLU, output 10, CrossEntropy")

if st.sidebar.button("학습 시작하기"):
    # 1) LLM → 스펙
    client = GPTClient()
    raw = client.to_graph_input(nl_desc)
    spec = raw
    st.subheader("파싱된 모델 스펙 (JSON)")
    st.json(spec)

    # 2) 모델 생성
    model = ModelBuilder.from_spec(spec, input_dim=784)
    st.subheader("생성된 모델 레이어")
    layer_info = [f"{i}: {type(l).__name__}" for i, l in enumerate(model.layers)]
    st.write("\n".join(layer_info))

    # 3) 데이터 로더 준비
    train_loader = MNISTLoader(train=True, batch_size=batch_size)
    val_loader = MNISTLoader(train=False, batch_size=100)

    # 4) 옵티마이저 & 트레이너 설정
    optimizer = Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, optimizer, train_loader, val_loader, num_epochs=epochs)

    # 5) 학습 진행 및 메트릭 수집
    loss_hist, acc_hist = [], []
    for ep in range(epochs):
        loss = trainer.train_one_epoch(ep)
        acc = trainer.validate_one_epoch(ep)
        loss_hist.append(loss)
        acc_hist.append(acc)

    # 6) 학습 곡선 시각화
    fig, ax = plt.subplots()
    ax.plot(loss_hist, label="훈련 손실")
    ax.plot(acc_hist, label="검증 정확도")
    ax.set_xlabel("에포크")
    ax.legend()
    st.pyplot(fig)

    # 7) 혼동 행렬 및 분류 리포트
    all_preds, all_labels = [], []
    for x, y in val_loader:
        out = trainer.model(x)
        preds = out.data.argmax(axis=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())
    cm = confusion_matrix(all_labels, all_preds)
    st.subheader("혼동 행렬")
    st.write(cm)
    st.subheader("분류 리포트")
    st.text(classification_report(all_labels, all_preds, digits=4))

# 안내 메시지
st.write("사이드바에서 하이퍼파라미터를 설정하고, 자연어로 모델을 정의한 뒤 '학습 시작하기' 버튼을 눌러보세요.")
