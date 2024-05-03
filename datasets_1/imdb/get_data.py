"""Implements dataloaders for IMDB dataset."""

from tqdm import tqdm
from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
import h5py
from gensim.models import KeyedVectors
# from .vgg import VGGClassifier
import os
import sys
from typing import *
import numpy as np

sys.path.append('/home/pliang/multibench/MultiBench/datasets/imdb')


class IMDBDataset(Dataset):
    """Implements a torch Dataset class for the imdb dataset."""
    
    def __init__(self, file: h5py.File, start_ind: int, end_ind: int, vggfeature: bool = False, img_feature=False) -> None:
        """Initialize IMDBDataset object.

        Args:
            file (h5py.File): h5py file of data
            start_ind (int): Starting index for dataset
            end_ind (int): Ending index for dataset
            vggfeature (bool, optional): Whether to return pre-processed vgg_features or not. Defaults to False.
        """
        self.file = file
        self.start_ind = start_ind
        self.size = end_ind-start_ind
        self.vggfeature = vggfeature
        self.img_feature = img_feature

    def __getitem__(self, ind):
        """Get item from dataset.

        Args:
            ind (int): Index of data to get

        Returns:
            tuple: Tuple of text input, image input, and label
        """
        if not hasattr(self, 'dataset'):
            self.dataset = h5py.File(self.file, 'r')
        text = self.dataset["features"][ind+self.start_ind]
        image = self.dataset["images"][ind+self.start_ind] if not self.vggfeature else \
            self.dataset["vgg_features"][ind+self.start_ind]
        label = self.dataset["genres"][ind+self.start_ind]

        if self.img_feature:
            full_image = self.dataset["images"][ind+self.start_ind]
            return text, image, full_image, label

        return text, image, label

    def __len__(self):
        """Get length of dataset."""
        return self.size


class IMDBDataset_robust(Dataset):
    """Implements a torch Dataset class for the imdb dataset that uses robustness measures as data augmentation."""

    def __init__(self, dataset, start_ind: int, end_ind: int) -> None:
        """Initialize IMDBDataset_robust object.

        Args:
            file (h5py.File): h5py file of data
            start_ind (int): Starting index for dataset
            end_ind (int): Ending index for dataset
            vggfeature (bool, optional): Whether to return pre-processed vgg_features or not. Defaults to False.
        """
        self.dataset = dataset
        self.start_ind = start_ind
        self.size = end_ind-start_ind

    def __getitem__(self, ind):
        """Get item from dataset.

        Args:
            ind (int): Index of data to get

        Returns:
            tuple: Tuple of text input, image input, and label
        """
        text = self.dataset[ind+self.start_ind][0]
        image = self.dataset[ind+self.start_ind][1]
        label = self.dataset[ind+self.start_ind][2]

        return text, image, label

    def __len__(self):
        """Get length of dataset."""
        return self.size


def _process_data(filename, path):
    data = {}
    filepath = os.path.join(path, filename)

    with Image.open(filepath+".jpeg") as f:
        image = np.array(f.convert("RGB"))
        data["image"] = image

    with open(filepath+".json", "r") as f:
        info = json.load(f)

        plot = info["plot"]
        data["plot"] = plot

    return data


def get_dataloader(path: str, test_path: str, num_workers: int = 8, train_shuffle: bool = True, batch_size: int = 40, vgg: bool = False, img_feature=False, skip_process=False, no_robust=False) -> Tuple[Dict]:
    """Get dataloaders for IMDB dataset.

    Args:
        path (str): Path to training datafile.
        test_path (str): Path to test datafile.
        num_workers (int, optional): Number of workers to load data in. Defaults to 8.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
        batch_size (int, optional): Batch size of data. Defaults to 40.
        vgg (bool, optional): Whether to return raw images or pre-processed vgg features. Defaults to False.
        skip_process (bool, optional): Whether to pre-process data or not. Defaults to False.
        no_robust (bool, optional): Whether to not use robustness measures as augmentation. Defaults to False.

    Returns:
        Tuple[Dict]: Tuple of Training dataloader, Validation dataloader, Test Dataloader
    """

    print(img_feature)

    train_dataloader = DataLoader(IMDBDataset(path, 0, 15552, vgg, img_feature),
                                  shuffle=train_shuffle, num_workers=num_workers, batch_size=batch_size)
    val_dataloader = DataLoader(IMDBDataset(path, 15552, 18160, vgg, img_feature),
                                shuffle=False, num_workers=num_workers, batch_size=batch_size)
    if no_robust:
        test_dataloader = DataLoader(IMDBDataset(path, 18160, 25959, vgg, img_feature),
                                     shuffle=False, num_workers=num_workers, batch_size=batch_size)
        return train_dataloader, val_dataloader, test_dataloader
