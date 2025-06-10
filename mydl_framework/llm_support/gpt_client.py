# mydl_framework/llm_support/gpt_client.py

import os
import openai
import json

class GPTClient:
    def __init__(self, api_key=None, model_name="gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key required")
        openai.api_key = self.api_key
        self.model_name = model_name

    def infer(self, prompt: str, **kwargs) -> str:
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content.strip()

    def to_graph_input(self, natural_language: str) -> dict:
        # 1) 프롬프트 생성
        prompt = (
            "Convert the following description into JSON spec for MyDL Framework models.\n"
            f"Description: '{natural_language}'\nJSON:"
        )
        # 2) LLM 호출 및 raw 응답
        raw = self.infer(prompt, max_tokens=500)
        print("<< RAW LLM RESPONSE >>")
        print(raw)

        # 3) JSON 본문 추출
        start = raw.find('{')
        end = raw.rfind('}')
        body = raw[start:end+1] if start != -1 and end != -1 else raw

        # 4) JSON 파싱
        spec = json.loads(body)

        # 5) 'model' 래핑 언랩
        if isinstance(spec.get("model"), dict):
            spec = spec["model"]

        # 6) 레이어 스펙 가공 (size 우선)
        layers = []
        for layer in spec.get("layers", []):
            raw_type = layer.get("type", "").lower()
            if raw_type == "input":
                continue

            # size → units → parameters 내부 키 순으로 추출
            size = layer.get("size")
            units = size or layer.get("units")
            params = layer.get("parameters", {}) or {}
            units = (
                units
                or params.get("units")
                or params.get("hidden_units")
                or params.get("output_units")
                or params.get("size")
            )
            activation = layer.get("activation") or params.get("activation")

            if raw_type in ("dense", "linear", "fullyconnected", "hidden", "output"):
                layers.append({"type": "Linear", "out_features": units})
                if activation and activation.lower() != "softmax":
                    layers.append({"type": activation})

            elif raw_type in ("relu", "sigmoid", "tanh"):
                layers.append({"type": layer.get("type")})

            else:
                layers.append(layer)

        # 7) 손실 함수 추출
        loss_field = spec.get("loss")
        if isinstance(loss_field, dict):
            loss_type = loss_field.get("type", "CrossEntropy")
        elif isinstance(loss_field, str):
            loss_type = loss_field
        else:
            loss_type = "CrossEntropy"

        return {"layers": layers, "loss": loss_type}
