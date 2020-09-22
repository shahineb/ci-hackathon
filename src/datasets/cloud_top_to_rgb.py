import os
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms


class CloudTOPtoRGBDataset(Dataset):

    def __init__(self, root):
        self.root = root
        self.transform = transforms.ToTensor()
        self._load_datasets()

    def _load_datasets(self):
        cloud_top_path = os.path.join(self.root, "X_train_CI20.npy")
        true_color_path = os.path.join(self.root, "Y_train_CI20.npy")
        self.cloud_top_dataset = np.load(cloud_top_path)
        self.true_color_dataset = np.load(true_color_path)
        ########
        threshold = 0.01
        fraction_of_null_pixels = np.all(self.true_color_dataset == 0., axis=-1).mean(axis=(1, 2))
        valid_samples = fraction_of_null_pixels < threshold
        self.cloud_top_dataset = self.cloud_top_dataset[valid_samples]
        self.true_color_dataset = self.true_color_dataset[valid_samples]
        ########

    def __getitem__(self, idx):
        # Load frames
        cloud_top = self.cloud_top_dataset[idx]
        true_color = self.true_color_dataset[idx]

        # Convert to tensors
        cloud_top = self.transform(cloud_top)
        true_color = self.transform(true_color)

        # Apply random flip augmentation
        if random.random() < 0.5:
            cloud_top = F.hflip(cloud_top)
            true_color = F.hflip(true_color)

        if random.random() < 0.5:
            cloud_top = F.vflip(cloud_top)
            true_color = F.vflip(true_color)
        return cloud_top, true_color

    def __len__(self):
        return len(self.cloud_top_dataset)

    @classmethod
    def build(cls, cfg):
        return cls(root=cfg['root'])
