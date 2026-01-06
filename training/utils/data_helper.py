import os
from .install_lib import install_package
install_package(["nibabel"])
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

def get_data_ids(root_dir):
    subject_ids = []
    for item_name in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item_name)
        if os.path.isdir(item_path):
            subject_ids.append(item_name)
    return subject_ids

class BRATSDataset2D(Dataset):
    def __init__(self, root_dir, subject_ids, slice_indices, transform=None):
        """
        root_dir: path containing one subfolder per subject
        subject_ids: list of folder names / prefixes
        slice_indices: list of z-indices to sample
        transform: callable(image, mask) -> (image, mask)
        """
        self.root   = root_dir
        self.ids    = subject_ids
        self.slices = slice_indices
        self.tfms   = transform
        self.mods   = ("t1c","t1n","t2f","t2w")

        # build list of valid (subject, slice) pairs where mask is non-empty
        self.samples = []
        for subj in self.ids:
            seg_path = os.path.join(self.root, subj, f"{subj}-t1n.nii.gz")
            seg_vol  = nib.load(seg_path).get_fdata()
            for z in self.slices:
                if np.any(seg_vol[:, :, z]):
                    self.samples.append((subj, z))

        self._cached_id   = None
        self._cached_vols = None

    def __len__(self):
        return len(self.samples)

    def _load_subject(self, subj):
        """Load all modality volumes for a subject into memory"""
        subj_folder = os.path.join(self.root, subj)
        vols = {}
        for mod in self.mods:
            fn   = f"{subj}-{mod}.nii.gz"
            path = os.path.join(subj_folder, fn)
            vols[mod] = nib.load(path).get_fdata().astype(np.float32)
        seg_fn  = f"{subj}-seg.nii.gz"
        seg_path= os.path.join(subj_folder, seg_fn)
        vols["seg"] = nib.load(seg_path).get_fdata().astype(np.int64)
        return vols

    def __getitem__(self, idx):
        subj, z = self.samples[idx]

        if subj != self._cached_id:
            self._cached_vols = self._load_subject(subj)
            self._cached_id   = subj
        vols = self._cached_vols

        image = np.stack([vols[mod][:, :, z] for mod in self.mods], axis=0)
        mask  = vols["seg"][:, :, z]

        if self.tfms:
            image, mask = self.tfms(image, mask)

        image = torch.from_numpy(image)
        mask  = torch.from_numpy(mask)

        return image, mask