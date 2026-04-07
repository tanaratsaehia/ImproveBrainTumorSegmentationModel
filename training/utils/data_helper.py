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
    

class UAVidDataset2D(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root = root_dir
        self.transform = transform
        self.num_classes = 8
        
        # Convert list to tensor for fast GPU/CPU distance calculation
        self.palette = torch.tensor([
            [128, 0, 0],      # 0: Building
            [128, 64, 128],   # 1: Road
            [0, 128, 0],      # 2: Tree
            [128, 128, 0],    # 3: Vegetation
            [64, 0, 128],     # 4: Moving Car
            [192, 0, 192],    # 5: Static Car
            [64, 64, 0],      # 6: Human
            [0, 0, 0]         # 7: Clutter
        ], dtype=torch.float32)

    def __len__(self):
        return len(self.df)
    
    def _rgb_to_index(self, mask_raw):
        # 1. If mask is already 2D (H, W)
        if len(mask_raw.shape) == 2:
            mask_idx = torch.from_numpy(mask_raw).long()
        else:
            # 2. If mask is 3D, handle shape and map colors
            if mask_raw.shape[0] == 3:
                mask_raw = np.transpose(mask_raw, (1, 2, 0))
            
            mask_tensor = torch.from_numpy(mask_raw).float()
            h, w, _ = mask_tensor.shape
            
            # Distance mapping logic
            mask_flat = mask_tensor.view(-1, 3)
            # Find the closest color in the palette for every pixel
            # (using cdist for efficiency on large images)
            distances = torch.cdist(mask_flat, self.palette.to(mask_flat.device)) 
            mask_idx = torch.argmin(distances, dim=1).view(h, w)

        # --- THE SAFETY CATCH ---
        # Force any pixel value to be within [0, 7]. 
        # If any pixel is 8, 255, etc., it becomes 7 (Clutter).
        mask_idx = torch.clamp(mask_idx, 0, self.num_classes - 1)
        
        return mask_idx.long()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load Image: already (3, 640, 640)
        image = np.load(os.path.join(self.root, row['img_path']))
        
        # Load Mask: if it was saved as RGB (H, W, 3) or (3, H, W)
        mask_raw = np.load(os.path.join(self.root, row['mask_path']))
        
        # Fix shape if it was saved as (3, H, W)
        if mask_raw.shape[0] == 3:
            mask_raw = np.transpose(mask_raw, (1, 2, 0))
            
        # Convert to 0-7 indices
        mask = self._rgb_to_index(mask_raw) # Returns a LongTensor (H, W)

        image = torch.from_numpy(image).float()

        # Apply transforms if any (Make sure they don't break the indices!)
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask