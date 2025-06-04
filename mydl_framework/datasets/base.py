import numpy as np

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr >= len(self.indices):
            raise StopIteration
        batch_indices = self.indices[self.ptr : self.ptr + self.batch_size]
        self.ptr += self.batch_size
        # 데이터셋에서 (image, label) 튜플을 뽑아서 배열로 만듦
        images = []
        labels = []
        for idx in batch_indices:
            img, lbl = self.dataset[idx]
            images.append(img.astype(np.float32) / 255.0)  # 예: 0~1 스케일링
            labels.append(lbl)
        # NumPy 배열로 합치기
        images = np.stack(images)  # (batch_size, H, W)
        labels = np.array(labels, dtype=np.int32)
        return images, labels
