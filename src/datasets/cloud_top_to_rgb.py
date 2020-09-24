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

        self.cloud_top_means = np.load("data_stats/means_CT_norm.npy")
        self.cloud_top_stds = np.load("data_stats/stds_CT_norm.npy")
        self.true_color_means = np.load("data_stats/means_TC_norm.npy")
        self.true_color_stds = np.load("data_stats/stds_TC_norm.npy")


    def __getitem__(self, idx):
        # Load frames
        cloud_top = self.cloud_top_dataset[idx]
        true_color = self.true_color_dataset[idx]

        ct_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=self.cloud_top_means, std=self.cloud_top_stds)])
        true_color_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=self.true_color_means, std=self.true_color_stds)])

        # Convert to tensor and normalise
        cloud_top = ct_transform(cloud_top)
        true_color = true_color_transform(true_color)

        # Apply random flip augmentation
        if random.random() < 0.5:
            cloud_top = F.hflip(cloud_top)
            true_color = F.hflip(true_color)

        if random.random() < 0.5:
            cloud_top = F.vflip(cloud_top)
            true_color = F.vflip(true_color)

        # Apply random crop (only if converted to PIL image - check Compose)
        #if random.random() < 0.5:
        #    cloud_top = F.RandomCrop(cloud_top)
        #    true_color = F.RandomCrop(true_color)

        return cloud_top, true_color


    def __len__(self):
        return len(self.cloud_top_dataset)

    @classmethod
    def build(cls, cfg):
        return cls(root=cfg['root'])
