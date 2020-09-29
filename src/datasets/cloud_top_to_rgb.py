import os
import random
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms


class CloudTOPtoRGBDataset(Dataset):
    """Dataset instance loading couple of top of cloud infrared image and
    its corresponding RGB optical counterpart

    Args:
        root (str): path to directory containing data
    """
    def __init__(self, root):
        self.root = root
        self.transform = transforms.ToTensor()
        self._load_datasets()

    def _load_datasets(self):
        cloud_top_path = os.path.join(self.root, "X_train_CI20.npy")
        true_color_path = os.path.join(self.root, "Y_train_CI20.npy")
        self.cloud_top_dataset = np.load(cloud_top_path)
        self.true_color_dataset = np.load(true_color_path)
        self._filter_black_frames()
        self._compute_image_statistics()

    def _filter_black_frames(self, threshold=0.99):
        """Drops frames out of the dataset if they have a percentage of
        black pixels greater than some threshold

        Args:
            threshold (float): greatest percentage of black pixels allowed

        """
        fraction_of_black_pixels = np.all(self.true_color_dataset == 0., axis=-1).mean(axis=(1, 2))
        valid_samples = fraction_of_black_pixels > threshold
        self.cloud_top_dataset = self.cloud_top_dataset[valid_samples]
        self.true_color_dataset = self.true_color_dataset[valid_samples]

    def _compute_image_statistics(self):
        self.mean_image = {'cloud_top': self.cloud_top_dataset.mean(axis=0),
                           'true_color': self.true_color_dataset.mean(axis=0)}
        self.std_image = {'cloud_top': self.cloud_top_dataset.std(axis=0).clip(min=np.finfo(np.float32).eps),
                          'true_color': self.true_color_dataset.std(axis=0).clip(min=np.finfo(np.float32).eps)}

    def __getitem__(self, idx):
        # Load frames
        cloud_top = self.cloud_top_dataset[idx]
        true_color = self.true_color_dataset[idx]

        # Normalize images
        cloud_top = (cloud_top - self.mean_image['cloud_top']) / self.std_image['cloud_top']
        true_color = (true_color - self.mean_image['true_color']) / self.std_image['true_color']

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
