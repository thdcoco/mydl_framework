import pytest
from testrix.llm_support.gpt_client import GPTClient
from testrix.llm_support.model_builder import ModelBuilder

def test_model_builder():
    spec = {"layers":[{"type":"Linear","out_features":10}],"loss":"CrossEntropy"}
    model = ModelBuilder.from_spec(spec, 5)
    assert len(model.layers) == 1