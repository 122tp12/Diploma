import torch
from torch.utils.data import Dataset
import os

class StrokeGraphDataset(Dataset):
    def __init__(self, root_dir, file_list):
        self.root_dir = root_dir
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        # Load the pre-processed Data object
        data = torch.load(file_path, weights_only=False)
        return data