# mydl_framework/datasets/mnist.py

import os
import gzip
import numpy as np

class MNIST:
    """
    MNIST 데이터셋 로더 클래스
    root: 데이터 파일들이 들어 있는 폴더 경로 (예: "./data")
    train: True이면 train 데이터, False이면 test 데이터 로드
    """
    def __init__(self, root, train=True):
        # 압축된 IDX 파일 이름
        if train:
            img_fname = "train-images-idx3-ubyte.gz"
            label_fname = "train-labels-idx1-ubyte.gz"
        else:
            img_fname = "t10k-images-idx3-ubyte.gz"
            label_fname = "t10k-labels-idx1-ubyte.gz"

        img_path = os.path.join(root, img_fname)
        label_path = os.path.join(root, label_fname)

        # 파일이 없으면 에러 발생
        if not os.path.exists(img_path) or not os.path.exists(label_path):
            raise FileNotFoundError(
                f"MNIST files not found in {root}. "
                f"Expected '{img_fname}' and '{label_fname}'."
            )

        # 이미지 / 라벨 로드
        self.images = self._load_images(img_path)
        self.labels = self._load_labels(label_path)

        # 데이터 개수가 이미지와 라벨에서 일치하는지 확인
        if len(self.images) != len(self.labels):
            raise RuntimeError("Mismatch between number of images and labels.")

    def __len__(self):
        # 총 샘플 개수 반환 (train=60000, test=10000)
        return len(self.labels)

    def __getitem__(self, index):
        # 인덱스를 주면 (이미지, 라벨) 튜플을 반환
        img = self.images[index]
        lbl = self.labels[index]
        return img, lbl

    @staticmethod
    def _load_images(img_gz_path):
        # gzip으로 압축된 IDX 이미지 파일 읽기
        with gzip.open(img_gz_path, "rb") as f:
            # IDX 헤더: magic number(32bit), 이미지 개수(32bit), 높이(32bit), 너비(32bit)
            magic = int.from_bytes(f.read(4), "big")
            if magic != 2051:
                raise RuntimeError(f"Invalid magic number {magic} in image file!")

            num_images = int.from_bytes(f.read(4), "big")
            rows = int.from_bytes(f.read(4), "big")
            cols = int.from_bytes(f.read(4), "big")

            # 남은 바이너리 데이터를 모두 읽어서 NumPy 배열로 변환
            buf = f.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols)
            return data

    @staticmethod
    def _load_labels(lbl_gz_path):
        # gzip으로 압축된 IDX 라벨 파일 읽기
        with gzip.open(lbl_gz_path, "rb") as f:
            # IDX 헤더: magic number(32bit), 라벨 개수(32bit)
            magic = int.from_bytes(f.read(4), "big")
            if magic != 2049:
                raise RuntimeError(f"Invalid magic number {magic} in label file!")

            num_labels = int.from_bytes(f.read(4), "big")

            # 남은 바이너리 데이터를 모두 읽어서 NumPy 배열으로 변환
            buf = f.read(num_labels)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return labels
