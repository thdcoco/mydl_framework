# mydl_framework/layers/base.py

import numpy as np
from itertools import chain
from mydl_framework import autodiff  # 순환 참조 방지를 위해 최상위 모듈 참조

class Layer:
    def __init__(self):
        # 학습 가능한 파라미터 변수(Variable)를 {name: Variable} 형태로 보관
        self.params = {}
        # 자식 레이어(Layer)를 {name: Layer} 형태로 보관
        self._layers = {}
        # train/eval 모드 구분 플래그
        self.training = True

    def __setattr__(self, name, value):
        """
        layer.foo = Bar() 식으로 레이어 속성에 Layer 인스턴스를 할당하면,
        자동으로 self._layers[name]에 등록한다.
        """
        # 'Layer' 클래스를 가져오기 위해, 불러오기에 순환 참조가 일어나지 않도록 최상위 모듈을 이용
        from mydl_framework.layers.base import Layer as _LayerBase

        if isinstance(value, _LayerBase):
            # 새로운 서브 레이어를 이름 name으로 등록
            self._layers[name] = value
        super().__setattr__(name, value)

    def forward(self, x):
        """
        각 Layer는 반드시 forward(x)를 구현해야 함.
        x: Variable 또는 ndarray
        """
        raise NotImplementedError()

    def __call__(self, x):
        """
        호출 시 자동으로 forward(x)를 실행하도록 함.
        """
        return self.forward(x)

    def parameters(self):
        """
        현재 레이어(self.params)와 모든 하위 레이어(self._layers)의
        params를 재귀적으로 순회하여 Variable 객체를 하나씩 yield.
        """
        # 1) 이 레이어에 속한 변수
        for _, v in self.params.items():
            yield v

        # 2) 자식 레이어(서브 레이어)들의 변수
        for _, layer in self._layers.items():
            yield from layer.parameters()

    def train(self):
        """
        현재 레이어 및 모든 하위 레이어들을 train 모드로 설정
        """
        self.training = True
        for _, layer in self._layers.items():
            layer.train()

    def eval(self):
        """
        현재 레이어 및 모든 하위 레이어들을 eval 모드로 설정
        """
        self.training = False
        for _, layer in self._layers.items():
            layer.eval()


class Model(Layer):
    def __init__(self):
        super().__init__()
        # Model을 자유롭게 구성할 때, add()를 통해 서브 레이어를 등록하거나
        # 속성 할당을 통해 자동으로 _layers에 등록 가능.
        # 예) self.l1 = Linear(...)
        #     self.add(SomeOtherLayer())
        # 그냥 빈 상태로 둬도 무방.

    def add(self, layer):
        """
        명시적으로 서브 레이어를 추가할 때 편리한 메서드.
        layer: Layer 인스턴스
        """
        layer_name = f"layer{len(self._layers)}"
        self._layers[layer_name] = layer

    def forward(self, x):
        """
        Model 클래스 자체에는 forward 구현이 없으며, 
        하위에서 반드시 override해야 함.
        """
        raise NotImplementedError()

    def __call__(self, x):
        """
        model(x) 형태로 호출 시 forward()가 실행됨.
        """
        return self.forward(x)
