testrix

testrix는 사용자가 GUI로 하이퍼파라미터를 조절하고, 자연어로 모델 구조를 정의하면 그 결과를 바탕으로 MNIST 분류 모델을 생성 → 학습 → 평가까지 한 번에 수행하는 실험 프레임워크입니다.

왼쪽 사이드바에서 학습률(learning rate), 배치 크기(batch size), 에포크(epochs), 은닉 유닛(hidden units) 등을 설정한 뒤 학습 시작하기 버튼을 누르면:

자연어 모델 설명을 JSON 스펙으로 파싱

JSON 스펙 기반으로 모델 자동 구성

학습 진행(손실/정확도 곡선) 시각화

혼동행렬(confusion matrix), 분류리포트(classification report), 샘플 예측까지 자동 출력

✨ Features

GUI 기반 하이퍼파라미터 설정

Learning rate / Batch size / Epochs / Hidden units 등

자연어 기반 모델 정의

입력한 문장을 JSON 스펙으로 파싱하여 모델 구성

자동 모델 빌드

Spec(JSON) → Layer 조합 → Model 생성

학습 결과 시각화

Loss / Accuracy curve 출력

평가 리포트 자동 생성

Confusion matrix

Classification report (precision/recall/F1)

Sample predictions

🧩 Architecture (High-level)

testrix는 아래 흐름으로 동작합니다.

GUI 입력

하이퍼파라미터 설정

자연어 모델 구조 입력

Spec Parser

자연어 → JSON 모델 스펙

Model Builder

JSON 스펙 → 레이어 구성 → 모델 생성

Trainer

MNIST 로딩 → 학습 → 평가

UI 출력

그래프/혼동행렬/분류리포트/샘플 예측 표시

🖥️ Screenshot (Optional)

레포에 스크린샷을 추가했다면 아래처럼 연결해두면 좋습니다.

<img src="assets/demo.png" width="900" alt="testrix demo" />

✅ Requirements

Python 3.9+

requirements.txt에 정의된 패키지들

⚙️ Installation
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

▶️ Run (GUI)
python app.py

🚀 Usage

python app.py 실행

사이드바에서 하이퍼파라미터 설정

learning rate / batch size / epochs / hidden units 등

모델 구조를 자연어로 입력

학습 시작하기 클릭

결과 확인

Training loss/accuracy curve

Confusion matrix

Classification report

Sample predictions

🧾 Model Spec (JSON)

testrix는 자연어로 입력된 모델 구조를 내부적으로 JSON 스펙으로 변환한 뒤, 해당 스펙을 기반으로 모델을 구성합니다.

아래는 사용자가 요청한 형태의 JSON 예시입니다. (GitHub/파서에서 바로 쓰기 좋게 문법을 정리했습니다.)

JSON example
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

Field guide

model.name
모델 이름(표시용)

model.layers
레이어 구성 리스트(순서대로 적용)

layers[].type
레이어 타입

hidden: 은닉층

output: 출력층

layers[].units
해당 레이어의 뉴런 수

layers[].activation (optional)
활성화 함수 이름 (예: ReLU, Sigmoid, Tanh 등)
보통 hidden 레이어에서 사용

model.loss_function
학습 손실 함수 (예: CrossEntropy)

참고: 실제 지원 키워드/옵션은 testfix/llm_support/ 및 예제(testfix/examples/)를 기준으로 확장할 수 있습니다.

🧪 Examples (CLI)

GUI 외에도 예제 스크립트로 빠르게 실행할 수 있습니다.

python -m testfix.examples.llm_mnist


디버그/튜닝용 예제:

python -m testfix.examples.llm_mnist_debug_tune

🧱 Project Structure
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

Module overview

testfix/autodiff : Variable/Function 기반 자동미분 코어

testfix/layers : Linear/Activation/Loss 등 레이어 구현

testfix/optimizers : SGD/Adam 최적화

testfix/training : 학습 루프(Trainer) 및 평가

testfix/datasets : MNIST 로더

testfix/llm_support : 자연어 파싱/스펙 생성/모델 빌더

testfix/examples : 실행 예제

tests : 유닛 테스트

✅ Testing
pytest -q

🛠️ Troubleshooting
MNIST 파일을 못 찾는 경우

data/ 폴더에 아래 4개 파일이 존재하는지 확인:

train-images-idx3-ubyte.gz

train-labels-idx1-ubyte.gz

t10k-images-idx3-ubyte.gz

t10k-labels-idx1-ubyte.gz

testfix/datasets/mnist_loader.py에서 경로가 data/로 맞는지 확인

자연어 파싱 결과가 이상한 경우

testfix/examples/llm_mnist_debug_tune.py로 스펙(JSON) 로그를 확인

testfix/llm_support/model_builder.py에서 스펙 해석 규칙 확인

🗺️ Roadmap (Optional)

 CNN 템플릿 지원(Conv/Pool 블록)

 Spec(JSON) 저장/불러오기(프리셋)

 실험 결과 export(JSON/CSV)

 Early stopping / LR scheduler

 모델 요약(파라미터 수/구조) UI 출력
