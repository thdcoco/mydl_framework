# mydl_framework/datasets/mnist_loader.py

import os
import gzip
import struct
import urllib.request
import numpy as np

# 데이터를 저장할 디렉터리
dir_data = './data'
os.makedirs(dir_data, exist_ok=True)

# MNIST 원본 URL
MNIST_URL = "http://yann.lecun.com/exdb/mnist"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz"
}

def download_and_parse(name: str):
    """.npy 파일이 없으면 원본 MNIST IDX를 내려받아 npy로 저장하고 리턴."""
    gz_name = FILES[name]
    gz_path = os.path.join(dir_data, gz_name)
    raw_path = gz_path[:-3]            # .gz 뗀 경로
    npy_path = os.path.join(dir_data, f"{name}.npy")

    # 이미 .npy가 있으면 바로 로드
    if os.path.exists(npy_path):
        return np.load(npy_path)

    # 없으면 .gz 다운로드
    if not os.path.exists(gz_path):
        print(f"Downloading {gz_name}...")
        urllib.request.urlretrieve(f"{MNIST_URL}/{gz_name}", gz_path)

    # 압축 해제
    with gzip.open(gz_path, 'rb') as f_in, open(raw_path, 'wb') as f_out:
        f_out.write(f_in.read())

    # IDX 포맷 파싱
    with open(raw_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic == 2051:  # image 파일
            rows, cols = struct.unpack(">II", f.read(8))
            buf = f.read(rows * cols * num)
            data = np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols)
        elif magic == 2049:  # label 파일
            buf = f.read(num)
            data = np.frombuffer(buf, dtype=np.uint8)
        else:
            raise ValueError(f"Unknown magic number {magic} in {raw_path}")

    # npy로 저장하고 cleanup
    np.save(npy_path, data)
    os.remove(raw_path)
    return data

class MNISTLoader:
    def __init__(self, train: bool = True, batch_size: int = 64):
        split = "train" if train else "test"
        imgs = download_and_parse(f"{split}_images")
        lbls = download_and_parse(f"{split}_labels")

        # reshape & 정규화
        self.images = imgs.reshape(-1, 28*28).astype(np.float32) / 255.0
        self.labels = lbls.astype(np.int64)

        self.batch_size = batch_size
        self.index = 0
        self.order = np.arange(len(self.labels))

    def __iter__(self):
        self.index = 0
        np.random.shuffle(self.order)
        return self

    def __next__(self):
        if self.index >= len(self.labels):
            raise StopIteration
        idx = self.order[self.index : self.index + self.batch_size]
        batch_x = self.images[idx]
        batch_y = self.labels[idx]
        self.index += self.batch_size
        return batch_x, batch_y
