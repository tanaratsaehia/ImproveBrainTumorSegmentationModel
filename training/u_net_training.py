import os
import sys
import torch
import argparse
import torch.nn as nn
import multiprocessing
import torch.optim as optim
from datetime import datetime

from training_helper import train_model, save_checkpoint
from data_helper import BRATSDataset2D, get_data_ids
from model_structure import UNet, UNetBiPyramid, HybridLoss
from torch.utils.data import DataLoader, random_split


parser = argparse.ArgumentParser(
    description="2D BraTS Segmentation Training Script."
)
parser.add_argument(
    '--data_dir',
    type=str,
    default='BraTS-10file-2per',
    help="Path to the root directory containing the BraTS datasets."
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=16,
    help=f"Batch size for DataLoader (default: 32)"
)
parser.add_argument(
    '--lr',
    type=float,
    default=4e-4,
    help=f"Learning rate (default: 1e-4)"
)
parser.add_argument(
    '--epochs',
    type=int,
    default=25,
    help=f"Number of training epochs (default: 1)"
)
parser.add_argument(
    '--val_split',
    type=float,
    default=0.2,
    help=f"Fraction of data to use for validation (default: 0.2)"
)
parser.add_argument(
    '--num_classes',
    type=int,
    default=4,
    help=f"Number of output classes (default: 4)"
)
parser.add_argument(
    '--resume',
    action='store_true',
    help="Resume training from the last saved checkpoint."
)

args = parser.parse_args()
SLICE_INDICES= list(range(150))

# Hyperparameters derived from arguments
ROOT_DIR     = os.path.join('BraTS-Datasets', args.data_dir)
BATCH_SIZE   = args.batch_size
LR           = args.lr
NUM_EPOCHS   = args.epochs
VAL_SPLIT    = args.val_split
NUM_CLASSES  = args.num_classes

NUM_WORKERS  = 2 # min(multiprocessing.cpu_count(), 4) if multiprocessing.cpu_count() > 1 else 0
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- Configuration ---")
print(f"Root Directory: {ROOT_DIR}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LR}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Validation Split: {VAL_SPLIT}")
print(f"Device: {DEVICE}")
print(f"Num Workers: {NUM_WORKERS}")
print(f"Num Classes: {NUM_CLASSES}")
print(f"---------------------")

if not os.path.isdir(ROOT_DIR):
    sys.exit(f"Error: Directory not found at {ROOT_DIR}")

SUBJECT_IDS = get_data_ids(ROOT_DIR)

full_dataset = BRATSDataset2D(
    root_dir    = ROOT_DIR,
    subject_ids = SUBJECT_IDS,
    slice_indices = SLICE_INDICES,
    transform   = None
)

print(f'Total data {len(full_dataset)} images.')

n_val = int(len(full_dataset) * VAL_SPLIT)
n_train = len(full_dataset) - n_val
train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)
val_loader = DataLoader(
    val_ds,   batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

model = UNet(in_channels=4, num_classes=NUM_CLASSES)
criterion = HybridLoss(NUM_CLASSES)
optimizer = optim.Adam(model.parameters(), lr=LR)
TRAIN_RESULT_PATH = 'training_results'
CHECKPOINT_DIR = os.path.join(TRAIN_RESULT_PATH, f'checkpoints_{model.model_name}')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_FILENAME = 'last_checkpoint.pth'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
start_epoch = 1

if args.resume and os.path.exists(CHECKPOINT_PATH):
    print(f"\nResuming training from {CHECKPOINT_PATH}...")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")
        start_epoch = 1

model.to(DEVICE)

report = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion, 
    optimizer=optimizer,
    num_epochs=NUM_EPOCHS,
    start_epoch=start_epoch,
    device=DEVICE,
    num_classes=NUM_CLASSES
)

final_completed_epoch = start_epoch + NUM_EPOCHS -1
save_checkpoint(
    model, 
    optimizer, 
    epoch=final_completed_epoch, 
    path=CHECKPOINT_PATH,
)

now = datetime.now()
timestamp = now.strftime("%d-%m-%Y_%H-%M-%S")
csv_file_name = f'{model.model_name}_{final_completed_epoch}epochs_{timestamp}.csv'
os.makedirs(os.path.join(TRAIN_RESULT_PATH, 'train_report'), exist_ok=True)
report.to_csv(os.path.join(TRAIN_RESULT_PATH, 'train_report', csv_file_name), index=False)

# BraTS-Datasets/BraTS-10file-2per
# BraTS-Datasets/BraTS-25file-5per