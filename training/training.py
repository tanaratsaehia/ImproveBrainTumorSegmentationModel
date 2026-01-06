import os
import sys
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from utils.training_helper import train_model, save_checkpoint
from utils.data_helper import BRATSDataset2D, get_data_ids
from model_structure import *
from torch.utils.data import DataLoader, random_split

start_running_time = time.time()

# ----------------------------------- Running Arguments -----------------------------------
parser = argparse.ArgumentParser(
    description="2D BraTS Segmentation Training Script."
)
parser.add_argument(
    'model_name',
    type=str,
    choices=["u_net", "u_net_se", "u_net_di", "u_net_se_di", "bipyramid", "bipyramid_se", "bipyramid_di", "bipyramid_se_di"],
    help="Name of the architecture to use. Options: %(choices)s"
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
    '--se_reduction',
    type=int,
    default=16,
    help="Squeeze and Excitation reduction rate."
)
parser.add_argument(
    '--dilation_rate',
    type=list,
    default=[1, 2, 1, 2],
    help="Dilation rate for u-net 4 layers."
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
    help=f"Number of training epochs (default: 25)"
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
parser.add_argument(
    '--patience',
    type=int,
    default=10,
    help="Patience for early stoping."
)

args = parser.parse_args()
SLICE_INDICES= list(range(150))

# Hyperparameters derived from arguments
MODEL_NAME    = args.model_name
SE_REDUCTION  = args.se_reduction
DILATION_RATE = args.dilation_rate
ROOT_DIR      = os.path.join('BraTS-Datasets', args.data_dir)
BATCH_SIZE    = args.batch_size
LR            = args.lr
NUM_EPOCHS    = args.epochs
VAL_SPLIT     = args.val_split
NUM_CLASSES   = args.num_classes
PATIENCE      = args.patience
NUM_WORKERS   = 2
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n\n----- Configuration -----")
print(f"Root Directory: {ROOT_DIR}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LR}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Patience: {PATIENCE}")
print(f"Validation Split: {VAL_SPLIT}")
print(f"Device: {DEVICE}")
print(f"Num Workers: {NUM_WORKERS}")
print(f"Num Classes: {NUM_CLASSES}")
print(f"Training device: {DEVICE}")



# ----------------------------------- Data preparation -----------------------------------
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
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

# ----------------------------------- Create Model -----------------------------------
model = None
if MODEL_NAME == "u_net":
    model = UNet(in_channels=4, num_classes=NUM_CLASSES)
elif MODEL_NAME == "u_net_se":
    model = UNetSE(in_channels=4, num_classes=NUM_CLASSES, 
                   reduction=SE_REDUCTION)
elif MODEL_NAME == "u_net_di":
    model = UNetDI(in_channels=4, num_classes=NUM_CLASSES, 
                   dilations_rate=DILATION_RATE)
elif MODEL_NAME == "u_net_se_di":
    model = UNetSeDi(in_channels=4, num_classes=NUM_CLASSES, 
                     reduction=SE_REDUCTION, dilations_rate=DILATION_RATE)
elif MODEL_NAME == "bipyramid":
    model = UNetBiPyramid(in_channels=4, num_classes=NUM_CLASSES)
elif MODEL_NAME == "bipyramid_se":
    model = UNetBiPyramidSE(in_channels=4, num_classes=NUM_CLASSES, 
                            reduction=SE_REDUCTION)
elif MODEL_NAME == "bipyramid_di":
    model = UNetBiPyramidDI(in_channels=4, num_classes=NUM_CLASSES, 
                            dilations_rate=DILATION_RATE)
elif MODEL_NAME == "bipyramid_se_di":
    model = UNetBiPyramidSeDi(in_channels=4, num_classes=NUM_CLASSES, 
                              reduction=SE_REDUCTION, dilations_rate=DILATION_RATE)
print(f"\nModel info: {model.model_info}")


criterion = HybridLoss(NUM_CLASSES)
optimizer = optim.Adam(model.parameters(), lr=LR)

TRAIN_RESULT_PATH = 'training_results'
CHECKPOINT_DIR = os.path.join(TRAIN_RESULT_PATH, f'checkpoints_{model.model_name}')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pth')
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'best_checkpoint.pth')
print(f"Model will save into: {CHECKPOINT_DIR}")
print(f"---------------------\n\n")

start_epoch = 1
if args.resume and os.path.exists(LAST_CHECKPOINT_PATH):
    print(f"\nResuming training from {LAST_CHECKPOINT_PATH}...")
    try:
        checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=DEVICE)
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

# ----------------------------------- Training -----------------------------------
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
    num_classes=NUM_CLASSES,
    start_run_time=start_running_time,
    best_save_path=BEST_CHECKPOINT_PATH,
    last_save_path=LAST_CHECKPOINT_PATH,
    patience=PATIENCE
    )

now = datetime.now()
timestamp = now.strftime("%d-%m-%Y_%H-%M-%S")
csv_file_name = f'{model.model_name}_{timestamp}.csv'
os.makedirs(os.path.join(TRAIN_RESULT_PATH, 'train_report'), exist_ok=True)
report.to_csv(os.path.join(TRAIN_RESULT_PATH, 'train_report', csv_file_name), index=False)

# BraTS-Datasets/BraTS-10file-2per
# BraTS-Datasets/BraTS-25file-5per