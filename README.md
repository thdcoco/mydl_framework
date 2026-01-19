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

## 🖼️ Screenshot (optional)

레포에 스크린샷 파일을 추가했다면 아래처럼 연결할 수 있습니다.

```md
<img src="assets/demo.png" width="900" alt="testrix demo" />


## ✅ Requirements

Python 3.9+

dependencies: requirements.txt 참고


## ⚙️ Installation

    git clone https://github.com/thdcoco/mydl_framework/tree/main/data
    cd https://github.com/thdcoco/mydl_framework/tree/main/data

    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate

    pip install -r requirements.txt


## 🚀 사용 방법 (Usage)
streamlit run app.py
왼쪽 사이드바에서 하이퍼파라미터 설정
learning rate / batch size / epochs / hidden units 등
모델 구조를 자연어로 입력
학습 시작하기 클릭
결과 확인

