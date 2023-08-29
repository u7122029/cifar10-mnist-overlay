import torch
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import Resize, Compose, ToTensor, ToPILImage
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
from torchvision.io import read_image
from torchvision.transforms.functional import resize


class CustomMNIST(Dataset):
    def __init__(self, data_path, label_path=None, transform=None):
        # aim: generate
        super().__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.data = np.load(self.data_path)
        self.labels = np.load(self.label_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        elem = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            elem = self.transform(elem)
        return elem, label


class ToNumpyRGB:
    def __init__(self):
        pass

    def __call__(self, sample):
        """

        :param sample: A PIL image.
        :return: A numpy array of shape (32,32,3)
        """
        sample = np.array(sample, dtype=np.uint8)
        return np.tile(sample,(3,1,1)).transpose(1,2,0)


def main():
    num_datasets = 1000

    upsize = Compose([Resize((32, 32)), ToNumpyRGB()])
    out_dir = Path("data") / "cifar10_overlayed"
    #dset_mnist = CustomMNIST(f"data/mnist-i/{i}/data.npy", f"data/mnist-i/{i}/labels.npy")
    dset_mnist = MNIST("/ml_datasets", train=False, transform=upsize)
    dset_cifar10 = CIFAR10("/ml_datasets", train=False)

    for dset_idx in tqdm(range(num_datasets)):
        out_data = []
        out_labels = []
        for img_idx in range(len(dset_mnist)):
            cifar_img_idx = torch.randint(low=0, high=len(dset_cifar10), size=(1,))[0].item()  # Pick random cifar10 image.
            cifar_img, cifar_label = dset_cifar10[cifar_img_idx]
            cifar_img = np.array(cifar_img)
            mnist_img, mnist_label = dset_mnist[img_idx]
            mnist_img = np.array(mnist_img) / 255
            cifar_img_inv = 255 - cifar_img

            result = np.array(np.round(mnist_img * cifar_img_inv + (1 - mnist_img) * cifar_img),dtype=np.uint8)
            out_data.append(np.array(result, dtype=np.uint8))
            out_labels.append(mnist_label)

        # save data (shape (10000,32,32,3))
        out_dset_root = Path(str(dset_idx).zfill(3))
        (out_dir / out_dset_root).mkdir(parents=True, exist_ok=True)
        out_full = np.array(out_data, dtype=np.uint8)
        assert out_full.shape == (10000,32,32,3)
        np.save(str(out_dir / out_dset_root / "data.npy"), out_full)
        np.save(str(out_dir / out_dset_root / "labels.npy"), np.array(out_labels, dtype=np.uint8))


if __name__ == "__main__":
    main()