# testrix

**testrix**는 사용자가 **GUI로 하이퍼파라미터를 조절**하고, **자연어로 모델 구조를 정의**하면 그 결과를 기반으로 **MNIST 분류 모델을 생성 · 학습 · 평가**까지 한 번에 수행하는 올인원 프레임워크입니다.

사이드바에서 학습률(learning rate), 배치 크기(batch size), 에포크(epochs), 은닉 유닛(hidden units) 등을 설정한 뒤 **학습 시작하기** 버튼을 누르면 다음이 자동으로 진행됩니다.

1. 자연어 모델 설명을 **모델 스펙(JSON)** 으로 파싱
2. 스펙에 맞춰 **모델 자동 구성**
3. **학습 진행(손실/정확도 곡선)** 시각화
4. 최종적으로 **혼동행렬(confusion matrix)**, **분류 리포트(classification report)**, **샘플 예측 결과** 출력

---

## ✨ Features

- **GUI 기반 하이퍼파라미터 설정**
  - learning rate / batch size / epochs / hidden units 등
- **자연어 기반 모델 정의**
  - 자연어 입력 → JSON 스펙 변환 → 모델 자동 구성
- **학습/평가 자동 파이프라인**
  - MNIST 로딩 → 학습 → 평가 → 결과 리포트 생성
- **시각화 & 리포트**
  - loss / accuracy curve
  - confusion matrix
  - classification report
  - sample predictions

---

## 🧠 How it works

testrix는 아래 파이프라인으로 동작합니다.

**GUI 입력** → **자연어 파싱** → **JSON 스펙 생성** → **모델 빌드** → **학습(Trainer)** → **평가/시각화 출력**

- **LLM/NL Parser**: 사용자의 자연어 입력을 구조화된 JSON으로 변환
- **Model Builder**: JSON 스펙 기반으로 레이어를 조합하여 모델 생성
- **Trainer**: 데이터 로딩, forward/backward, optimizer step, metric logging 수행
- **UI Renderer**: 그래프/리포트/샘플 예측 결과 표시

---


## ✅ Requirements

Python 3.9+

dependencies: requirements.txt 참고

---

## ⚙️ Installation

    git clone https://github.com/thdcoco/mydl_framework/tree/main/data
    cd https://github.com/thdcoco/mydl_framework/tree/main/data

    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate

    pip install -r requirements.txt

---

## 🚀 사용 방법 (Usage)

    streamlit run app.py
    
왼쪽 사이드바에서 하이퍼파라미터 설정
learning rate / batch size / epochs / hidden units 등
모델 구조를 자연어로 입력
학습 시작하기 클릭
결과 확인

---

## 🧾 Model Spec (JSON)

testrix는 자연어로 입력된 모델 구조를 내부적으로 JSON 스펙으로 변환한 뒤, 해당 스펙을 기반으로 모델을 구성합니다.

JSON 예시

아래는 3-Layer MLP를 정의하는 JSON 예시입니다.

    {
      "model": {
        "name": "3-Layer MLP",
        "layers": [
          {
            "type": "hidden",
            "units": 256,
            "activation": "ReLU"
          },
          {
            "type": "output",
            "units": 10
          }
        ],
        "loss_function": "CrossEntropy"
      }
    }

필드 설명 (Field Guide)

model.name
모델 이름(표시용)

model.layers
레이어 구성 리스트(앞에서부터 순서대로 적용)

layers[].type
레이어 타입

hidden: 은닉층

output: 출력층

layers[].units
해당 레이어의 뉴런 수

layers[].activation (optional)
활성화 함수 이름 (예: ReLU, Sigmoid, Tanh 등)
일반적으로 hidden 레이어에서 사용

model.loss_function
손실 함수 (예: CrossEntropy)

참고: 실제 지원하는 스펙 키워드/옵션은 testfix/llm_support/ 및 예제(testfix/examples/)를 기준으로 확장할 수 있습니다.

---

## 🧱 Project Structure
    .
    ├─ data/
    │  ├─ t10k-images-idx3-ubyte.gz
    │  ├─ t10k-labels-idx1-ubyte.gz
    │  ├─ train-images-idx3-ubyte.gz
    │  └─ train-labels-idx1-ubyte.gz
    ├─ testfix/
    │  ├─ autodiff/
    │  │  ├─ __init__.py
    │  │  ├─ function.py
    │  │  └─ variable.py
    │  ├─ datasets/
    │  │  ├─ __init__.py
    │  │  └─ mnist_loader.py
    │  ├─ examples/
    │  │  ├─ __init__.py
    │  │  ├─ llm_mnist.py
    │  │  └─ llm_mnist_debug_tune.py
    │  ├─ layers/
    │  │  ├─ __init__.py
    │  │  ├─ activations.py
    │  │  ├─ base.py
    │  │  ├─ linear.py
    │  │  └─ softmax_cross_entropy.py
    │  ├─ llm_support/
    │  │  ├─ __init__.py
    │  │  ├─ gpt_client.py
    │  │  ├─ loss.py
    │  │  └─ model_builder.py
    │  ├─ optimizers/
    │  │  ├─ __init__.py
    │  │  ├─ adam.py
    │  │  └─ sgd.py
    │  ├─ training/
    │  │  ├─ __init__.py
    │  │  └─ trainer.py
    │  └─ __init__.py
    ├─ tests/
    │  ├─ test_autodiff_core.py
    │  ├─ test_datasets_mnist.py
    │  └─ test_llm_support.py
    ├─ .gitignore
    ├─ README.md
    ├─ app.py
    ├─ requirements.txt
    └─ setup.py

모듈 요약 (Modules)

testfix/autodiff
Variable/Function 기반 자동미분 코어

testfix/layers
Linear/Activation/Loss 등 레이어 구현

testfix/optimizers
SGD / Adam 최적화 알고리즘

testfix/training
학습 루프(Trainer) 및 평가 로직

testfix/datasets
MNIST 데이터 로더

testfix/llm_support
자연어 → JSON 스펙 변환 및 모델 빌더

testfix/examples
실행 예제

tests
유닛 테스트


---


## 🛠️ Troubleshooting
MNIST 파일을 못 찾는 경우

data/ 폴더에 아래 4개 파일이 있는지 확인하세요.

train-images-idx3-ubyte.gz

train-labels-idx1-ubyte.gz

t10k-images-idx3-ubyte.gz

t10k-labels-idx1-ubyte.gz

자연어 파싱 결과가 이상한 경우

testfix/examples/llm_mnist_debug_tune.py로 JSON 스펙 출력/로그를 확인하세요.

testfix/llm_support/model_builder.py의 스펙 처리 규칙을 확인하세요.

---

## 🗺️ Roadmap (optional)

 CNN 템플릿 지원(Conv/Pool 블록)

 Spec(JSON) 저장/불러오기(프리셋)

 실험 결과 export(JSON/CSV)

 Early stopping / LR scheduler

 모델 요약(파라미터 수/구조) UI 출력
