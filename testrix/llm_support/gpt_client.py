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
        prompt = (
            "Convert the following description into JSON spec for MyDL Framework models.\n"
            f"Description: '{natural_language}'\nJSON:"
        )
        raw = self.infer(prompt, max_tokens=500)
        print("<< RAW LLM RESPONSE >>")
        print(raw)

        # JSON body만 꺼내 파싱
        start = raw.find('{')
        end   = raw.rfind('}')
        body  = raw[start:end+1] if start != -1 and end != -1 else raw
        spec  = json.loads(body)

        # 언랩 model 래핑
        if isinstance(spec.get("model"), dict):
            spec = spec["model"]

        # 손실 추출
        loss_field = spec.get("loss") or spec.get("loss_function")
        loss_type  = (
            loss_field.get("type") if isinstance(loss_field, dict) 
            else (loss_field if isinstance(loss_field, str) else "CrossEntropy")
        )

        # 레이어 가공
        layers = []
        for layer in spec.get("layers", []):
            t = layer.get("type", "").lower()
            # input 스킵
            if "input" in t:
                continue
            # loss name 스킵
            if loss_type.lower() in t:
                continue

            # units, activation 획득
            units      = layer.get("units") or layer.get("size")
            params     = layer.get("parameters", {}) or {}
            units     = units or params.get("units") or params.get("hidden_units") or params.get("output_units")
            activation = layer.get("activation") or params.get("activation")

            # dense가 포함된 모든 타입 → Linear
            if "dense" in t or "fullyconnected" in t or "hidden" in t or "output" in t:
                layers.append({"type": "Linear", "out_features": units})
                # Softmax는 손실 함수에 포함하므로 생략
                if activation and activation.lower() != "softmax":
                    layers.append({"type": activation})
            # activation 단독
            elif t in ("relu", "sigmoid", "tanh"):
                layers.append({"type": layer.get("type")})
            # 기타 그대로
            else:
                layers.append(layer)

        return {"layers": layers, "loss": loss_type}
