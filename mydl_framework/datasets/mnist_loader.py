# mydl_framework/datasets/mnist_loader.py

import os
import numpy as np

# 데이터를 저장/로드할 기본 디렉터리
dir_data = './data'

class MNISTLoader:
    def __init__(self, train=True, batch_size=64):
        img_path = os.path.join(dir_data, f'{"train" if train else "test"}_images.npy')
        lbl_path = os.path.join(dir_data, f'{"train" if train else "test"}_labels.npy')

        if os.path.exists(img_path) and os.path.exists(lbl_path):
            self.images = np.load(img_path)
            self.labels = np.load(lbl_path)
        else:
            # 파일이 없으면 테스트용 더미 데이터 생성
            n = 100
            self.images = np.random.randint(0, 256, size=(n, 28, 28), dtype=np.uint8)
            self.labels = np.random.randint(0, 10, size=(n,), dtype=np.int64)

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
        idx = self.order[self.index:self.index + self.batch_size]
        batch_x = self.images[idx].reshape(-1, 784) / 255.0
        batch_y = self.labels[idx]
        self.index += self.batch_size
        return batch_x, batch_y
