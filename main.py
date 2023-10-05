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
import random


class CustomMNIST(Dataset):
    def __init__(self, data_path, label_path=None, transform=None):
        # aim: generate
        super().__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.data = np.load(self.data_path)
        self.labels = np.load(self.label_path)
        print(self.data.shape)

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


def inverse_overlay():
    num_datasets = 1000

    upsize = Compose([Resize((32, 32)), ToNumpyRGB()])
    out_dir = Path("data") / "cifar10_overlayed"
    dset_mnist = CustomMNIST(f"data/mnist-i/{i}/data.npy", f"data/mnist-i/{i}/labels.npy")
    #dset_mnist = MNIST("/ml_datasets", train=False, transform=upsize)
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


def superimpose(data_dir: Path, idx):
    # Place digit right on top of cifar10 image. Digit is of randomised colour.
    # Intensity of mnist pixel determines how close that pixel will be to the randomised colour.
    num_datasets = 1000

    upsize = Compose([ToPILImage(), Resize((32, 32)), ToNumpyRGB()])
    out_dir = Path("data") / "cifar10_superimposed"
    dset_mnist = CustomMNIST(str(data_dir / "data.npy"), str(data_dir / "labels.npy"))
    #dset_mnist = MNIST("/ml_datasets", train=False, transform=upsize)
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
            solid_colour = np.ones((32,32,3)) * np.random.randint(0,256,3)

            result = np.array(np.round(mnist_img * solid_colour + (1 - mnist_img) * cifar_img),dtype=np.uint8)
            out_data.append(np.array(result, dtype=np.uint8))
            out_labels.append(mnist_label)

        # save data (shape (10000,32,32,3))
        out_dset_root = Path(str(dset_idx).zfill(3))
        (out_dir / str(idx) / out_dset_root).mkdir(parents=True, exist_ok=True)
        out_full = np.array(out_data, dtype=np.uint8)
        assert out_full.shape == (10000,32,32,3)
        np.save(str(out_dir / str(idx) / out_dset_root / "data.npy"), out_full)
        np.save(str(out_dir / str(idx) / out_dset_root / "labels.npy"), np.array(out_labels, dtype=np.uint8))


def colour_assign(data_dir: str, idx):
    # Set background intensity (0) to colour A, set digit intensity (1) to colour B.
    # Any pixel whose intensity is between 0 and 1 is assigned a colour between A and B where the two colours are
    # connected together in RGB space by a straight line, and the intensity determines how far along the line we choose
    # this colour.
    num_datasets = 1000

    upsize = Compose([Resize((32, 32)), ToNumpyRGB()])
    out_dir = Path("data") / "coloured"
    #dset_mnist = CustomMNIST(str(data_dir / "data.npy"), str(data_dir / "labels.npy"))
    dset_mnist = MNIST(data_dir, train=False, transform=upsize)
    #dset_cifar10 = CIFAR10("/ml_datasets", train=False)
    # 95% from List of 20 Simple, Distinct Colors - Sacha Trubetskoy.
    colours = ['(230, 25, 75)', '(60, 180, 75)', '(255, 225, 25)', '(0, 130, 200)', '(245, 130, 48)', '(145, 30, 180)',
               '(70, 240, 240)', '(240, 50, 230)', '(210, 245, 60)', '(250, 190, 212)', '(0, 128, 128)',
               '(220, 190, 255)', '(170, 110, 40)', '(255, 250, 200)', '(128, 0, 0)', '(170, 255, 195)',
               '(128, 128, 0)', '(255, 215, 180)', '(0, 0, 128)', '(128, 128, 128)', '(255, 255, 255)', '(0, 0, 0)']
    colours = set([eval(x) for x in colours])

    for dset_idx in tqdm(range(num_datasets)):
        out_data = []
        out_labels = []
        for img_idx in range(len(dset_mnist)):
            high_rgb = random.choice(tuple(colours))
            remaining_colours = colours - {high_rgb}
            low_rgb = random.choice(tuple(remaining_colours))
            high_intensity = np.ones((32,32,3)) * np.array(high_rgb)
            low_intensity = np.ones((32,32,3)) * np.array(low_rgb)

            mnist_img, mnist_label = dset_mnist[img_idx]
            mnist_img = np.array(mnist_img) / 255

            result = np.array(np.round(mnist_img * high_intensity + (1 - mnist_img) * low_intensity),dtype=np.uint8)
            out_data.append(np.array(result, dtype=np.uint8))
            out_labels.append(mnist_label)

        # save data (shape (10000,32,32,3))
        out_dset_root = Path(str(dset_idx).zfill(3))
        (out_dir / str(idx) / out_dset_root).mkdir(parents=True, exist_ok=True)
        out_full = np.array(out_data, dtype=np.uint8)
        assert out_full.shape == (10000,32,32,3)
        np.save(str(out_dir / str(idx) / out_dset_root / "data.npy"), out_full)
        np.save(str(out_dir / str(idx) / out_dset_root / "labels.npy"), np.array(out_labels, dtype=np.uint8))


if __name__ == "__main__":
    colour_assign("C:/ml_datasets", 0)