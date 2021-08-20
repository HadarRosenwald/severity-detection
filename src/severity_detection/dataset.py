
"""
Here, we create a custom dataset
"""
import torch
import pickle

from utils.types import PathT
from torch.utils.data import Dataset
import torchxrayvision as xrv
from typing import Any, Tuple, Dict, List


class MyDataset(Dataset):
    """
    Custom dataset template. Implement the empty functions.
    """
    def __init__(self, image_path: PathT, csv_path: PathT) -> None:
        # Set variables
        self.image_path = image_path
        self.csv_path = csv_path

        # Load features
        self.features = self._get_features()

        # Create list of entries
        self.entries = self._get_entries()

    def __getitem__(self, index: int) -> Tuple:
        return self.entries[index]['x'], self.entries[index]['y']

    def __len__(self) -> int:
        """
        :return: the length of the dataset (number of sample).
        """
        return len(self.entries)

    def _get_features(self) -> Any:
        """
        Load all features into a structure (not necessarily dictionary). Think if you need/can load all the features
        into the memory.
        :return:
        :rtype:
        """
        features = xrv.datasets.COVID19_Dataset(imgpath=self.image_path, csvpath=self.csv_path)

        # with open(self.path, "rb") as features_file:
        #     features = pickle.load(features_file)

        return features

    def _get_entries(self) -> List:
        """
        This function create a list of all the entries. We will use it later in __getitem__
        :return: list of samples
        """
        entries = []

        for item in self.features:
            entries.append(self._get_entry(item))

        return entries

    @staticmethod
    def _get_entry(item: Dict) -> Dict:
        """
        :item: item from the data. In this example, {'input': Tensor, 'y': int}
        """
        x = item['img'][0]
        y = torch.Tensor([sum(item['lab']), 0]) # todo - fix label

        return {'x': x, 'y': y}