import pytest
from testrix.datasets.mnist_loader import MNISTLoader

def test_mnist_loader():
    loader = MNISTLoader(train=True, batch_size=32)
    batches = list(iter(loader))
    assert len(batches[0]) == 2
    x, y = batches[0]
    assert x.shape[1] == 784