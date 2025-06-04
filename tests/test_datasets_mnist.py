# tests/test_datasets_mnist.py

import os
import pytest
import numpy as np
from mydl_framework.datasets.mnist import MNIST

@pytest.fixture(scope="module")
def mnist_data_dir():
    # 프로젝트 루트의 data/ 폴더에 MNIST 파일이 있다고 가정
    return os.path.join(os.getcwd(), "data")

def test_mnist_length_and_items(mnist_data_dir):
    # Train 데이터셋 로드 테스트
    train_ds = MNIST(root=mnist_data_dir, train=True)
    assert len(train_ds) == 60000
    x0, y0 = train_ds[0]
    assert isinstance(x0, np.ndarray)
    assert x0.shape == (28, 28)
    assert isinstance(y0, np.integer)

    # Test 데이터셋 로드 테스트
    test_ds = MNIST(root=mnist_data_dir, train=False)
    assert len(test_ds) == 10000
    x1, y1 = test_ds[9999]
    assert isinstance(x1, np.ndarray)
    assert x1.shape == (28, 28)
    assert isinstance(y1, np.integer)

def test_mnist_file_not_found(tmp_path):
    # 잘못된 경로를 주면 FileNotFoundError 발생
    with pytest.raises(FileNotFoundError):
        MNIST(root=str(tmp_path), train=True)
