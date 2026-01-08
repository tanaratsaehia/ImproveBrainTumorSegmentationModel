import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class BRATSDataset2D(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        """
        csv_path: path to the dataset_map.csv generated above
        root_dir: path to the preprocessed folder
        """
        self.df = pd.read_csv(csv_path)
        self.root = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image = np.load(os.path.join(self.root, row['img_path']))
        mask = np.load(os.path.join(self.root, row['mask_path']))

        image = image.astype(np.float32)
        mask = mask.astype(np.int64)

        if self.transform:
            image, mask = self.transform(image, mask)

        return torch.from_numpy(image), torch.from_numpy(mask)