import os
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from pathlib import Path

def preprocess_to_2d(root_dir, output_dir):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    
    mods = ("t1c", "t1n", "t2f", "t2w")
    data_records = []
    
    subject_ids = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(f"Found {len(subject_ids)} subjects. Starting extraction...")
    total_slice = 0

    for subj in subject_ids:
        subj_folder = os.path.join(root_dir, subj)
        
        vols = {m: nib.load(os.path.join(subj_folder, f"{subj}-{m}.nii.gz")).get_fdata().astype(np.float32) for m in mods}
        mask_vol = nib.load(os.path.join(subj_folder, f"{subj}-seg.nii.gz")).get_fdata().astype(np.uint8)
        
        depth = mask_vol.shape[2]
        total_slice += depth
        for z in range(depth):
            mask_slice = mask_vol[:, :, z]
            confirm_vols = vols[mods[1]][:, :, z]
            
            if np.any(confirm_vols > 0):
                img_slice = np.stack([vols[m][:, :, z] for m in mods], axis=0)
                
                img_name = f"{subj}_s{z}.npy"
                mask_name = f"{subj}_s{z}_mask.npy"
                
                img_path = os.path.join("images", img_name)
                mask_path = os.path.join("masks", mask_name)
                
                np.save(os.path.join(output_dir, img_path), img_slice)
                np.save(os.path.join(output_dir, mask_path), mask_slice)
                
                data_records.append({
                    "subject": subj,
                    "slice": z,
                    "img_path": img_path,
                    "mask_path": mask_path
                })

    df = pd.DataFrame(data_records)
    df.to_csv(os.path.join(output_dir, "dataset_mapper.csv"), index=False)
    print(f"Total slice found: {total_slice}")
    print(f"Preprocessing complete. Total slices saved: {len(df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2D BraTS Segmentation Data Slicing."
    )
    parser.add_argument(
        'raw_3d_path',
        type=str,
        help="Original 3D file (.nii.gz) path."
    )
    parser.add_argument(
        'output_path',
        type=str,
        help="Path to save sliced images."
    )
    args = parser.parse_args()
    original_path = args.raw_3d_path
    original_folder_name = Path(original_path).name
    output_path  = os.path.join(args.output_path, f"SLICED_{original_folder_name}")
    print(f"Slice data from: {original_path}")
    print(f"Save at: {output_path}")
    preprocess_to_2d(original_path, output_path)

# ./BraTS-Datasets/Trainset BraTS-Datasets