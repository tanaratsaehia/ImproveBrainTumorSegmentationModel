import gc
import os
import sys
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

import mlflow
import mlflow.pytorch
from mlflow.artifacts import download_artifacts
from dotenv import load_dotenv

from utils.training_helper import train_model, save_checkpoint
from utils.data_helper import BRATSDataset2D
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
    choices=["u_net", "u_net_se", "u_net_di", "u_net_se_di", "u_net_ag", "u_net_aspp", 
             "u_net_ag_aspp", "u_net_res", "u_net_res_4layer", "u_net_hybrid", "bipyramid", 
             "bipyramid_se", "bipyramid_di", "bipyramid_se_di", "u_net_4layer", "u_net_shadow_4layer", 
             "u_net_shadow_base32", "u_net_shadow_full", "u_net_dense_aspp", "u_net_scse"],
    help="Name of the architecture to use. Options: %(choices)s"
)
parser.add_argument(
    '--optim',
    default="adam_w",
    type=str,
    choices=["adam", "adam_w"],
    help="Name of the optimizer to use. Options: %(choices)s"
)
parser.add_argument(
    '--train_data_dir',
    type=str,
    default='SLICED_Trainset',
    help="Path to the train directory containing the BraTS datasets."
)
parser.add_argument(
    '--val_data_dir',
    type=str,
    default='SLICED_Valset',
    help="Path to the validate directory containing the BraTS datasets."
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=16,
    help=f"Batch size for DataLoader (default: 16)"
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
    default=2e-4,
    help=f"Learning rate (default: 2e-4)"
)
parser.add_argument(
    '--epochs',
    type=int,
    default=125,
    help=f"Number of training epochs (default: 100)"
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
    '--load_best',
    action='store_true',
    help="For resume training from the best saved checkpoint."
)
parser.add_argument(
    '--patience',
    type=int,
    default=25,
    help="Patience for early stoping (default 5)."
)
parser.add_argument(
    '--loss_ce_weight',
    type=float,
    default=0.5,
    help="Crossentropy loss weight (default 0.5)."
)
parser.add_argument(
    '--loss_dice_weight',
    type=float,
    default=0.5,
    help="Dice loss weight (default 0.5)."
)

args = parser.parse_args()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Hyperparameters derived from arguments
MODEL_NAME     = args.model_name
OPTIMIZER_NAME = args.optim
SE_REDUCTION   = args.se_reduction
DILATION_RATE  = args.dilation_rate
TRAIN_DATA_DIR = os.path.join('BraTS-Datasets', args.train_data_dir)
VAL_DATA_DIR   = os.path.join('BraTS-Datasets', args.val_data_dir)
BATCH_SIZE     = args.batch_size
LR             = args.lr
NUM_EPOCHS     = args.epochs
LOSS_CE_WEIGHT   = args.loss_ce_weight
LOSS_DICE_WEIGHT = args.loss_dice_weight
NUM_CLASSES   = args.num_classes
PATIENCE      = args.patience
NUM_WORKERS   = 2
LOAD_BEST     = args.load_best
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n\n----- Configuration -----")
print(f"Train data directory: {TRAIN_DATA_DIR}")
print(f"Validate data directory: {VAL_DATA_DIR}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Learning Rate: {LR}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Patience: {PATIENCE}")
print(f"Crossentropy loss weight: {LOSS_CE_WEIGHT}")
print(f"Dice loss weight: {LOSS_DICE_WEIGHT}")
print(f"Num Workers: {NUM_WORKERS}")
print(f"Num Classes: {NUM_CLASSES}")
print(f"Training device: {DEVICE}")

# ----------------------------------- Data preparation -----------------------------------
if not os.path.isdir(TRAIN_DATA_DIR) and not os.path.isdir(VAL_DATA_DIR):
    sys.exit(f"Error: Directory not found at train or validate")

train_dataset = BRATSDataset2D(
    csv_path    = os.path.join(TRAIN_DATA_DIR, 'dataset_mapper.csv'),
    root_dir    = TRAIN_DATA_DIR,
    transform   = None
)
val_dataset = BRATSDataset2D(
    csv_path    = os.path.join(VAL_DATA_DIR, 'dataset_mapper.csv'),
    root_dir    = VAL_DATA_DIR,
    transform   = None
)

print(f'Total data {len(train_dataset) + len(val_dataset)} images.')
print(f"Train: {len(train_dataset)} imgs | Validate: {len(val_dataset)} imgs")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=False
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=False
)

# ----------------------------------- Create Model -----------------------------------
model = None
if MODEL_NAME == "u_net":
    model = UNet(in_channels=4, num_classes=NUM_CLASSES)
elif MODEL_NAME == "u_net_4layer":
    model = UNet4Layer(in_channels=4, num_classes=NUM_CLASSES)
elif MODEL_NAME == "u_net_se":
    model = UNetSE(in_channels=4, num_classes=NUM_CLASSES, 
                   reduction=SE_REDUCTION)
elif MODEL_NAME == "u_net_di":
    model = UNetDI(in_channels=4, num_classes=NUM_CLASSES, 
                   dilations_rate=DILATION_RATE)
elif MODEL_NAME == "u_net_se_di":
    model = UNetSeDi(in_channels=4, num_classes=NUM_CLASSES, 
                     reduction=SE_REDUCTION, dilations_rate=DILATION_RATE)
elif MODEL_NAME == "u_net_ag":
    model = UNetAG(in_channels=4, num_classes=NUM_CLASSES)
elif MODEL_NAME == "u_net_aspp":
    model = UNetASPP(in_channels=4, num_classes=NUM_CLASSES)
elif MODEL_NAME == "u_net_hybrid":
    model = HybridUNet(in_channels=4, num_classes=NUM_CLASSES)
elif MODEL_NAME == "u_net_ag_aspp":
    model = UNetAG_ASPP(in_channels=4, num_classes=NUM_CLASSES)
elif MODEL_NAME == "u_net_res":
    model = UNetRes(in_channels=4, num_classes=NUM_CLASSES)
elif MODEL_NAME == "u_net_res_4layer":
    model = UNetRes4Layer(in_channels=4, num_classes=NUM_CLASSES)

elif MODEL_NAME == "bipyramid":
    model = UNetBiPyramid(in_channels=4, num_classes=NUM_CLASSES, deep_supervision=True)
elif MODEL_NAME == "bipyramid_se":
    model = UNetBiPyramidSE(in_channels=4, num_classes=NUM_CLASSES, 
                            reduction=SE_REDUCTION)
elif MODEL_NAME == "bipyramid_di":
    model = UNetBiPyramidDI(in_channels=4, num_classes=NUM_CLASSES, 
                            dilations_rate=DILATION_RATE)
elif MODEL_NAME == "bipyramid_se_di":
    model = UNetBiPyramidSeDi(in_channels=4, num_classes=NUM_CLASSES, 
                              reduction=SE_REDUCTION, dilations_rate=DILATION_RATE)

elif MODEL_NAME == "u_net_shadow_4layer":
    model = ParallelShadowUNet4Layer(in_channels=4, num_classes=NUM_CLASSES)
elif MODEL_NAME == "u_net_shadow_base32":
    model = ParallelShadowUNetbase32(in_channels=4, num_classes=NUM_CLASSES)
elif MODEL_NAME == "u_net_shadow_full":
    model = ParallelShadowUNet(in_channels=4, num_classes=NUM_CLASSES)

# "u_net_dense_aspp", "u_net_scse"
elif MODEL_NAME == "u_net_dense_aspp":
    model = UNetDenseASPP(in_channels=4, num_classes=NUM_CLASSES)
elif MODEL_NAME == "u_net_scse":
    model = UNet_scSE(in_channels=4, num_classes=NUM_CLASSES)
else:
    print("ERROR: Model name miss match!")
    time.sleep(10)
    os._exit(0)
print(f"\nModel info: {model.model_info}")
model.to(DEVICE)

if OPTIMIZER_NAME == "adam":
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif OPTIMIZER_NAME == "adam_w":
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
else:
    print(f"Unkonw optimizer '{OPTIMIZER_NAME}'")
    sys.exit(1)

criterion = HybridLoss(NUM_CLASSES, ce_weight=LOSS_CE_WEIGHT, dice_weight=LOSS_DICE_WEIGHT)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max', # max mode for dice | min mode for loss
    factor=0.5,
    patience=10,
    min_lr=1e-6
) 

TRAIN_RESULT_PATH = 'training_results'
CHECKPOINT_DIR = os.path.join(TRAIN_RESULT_PATH, f'checkpoints_{model.model_name}')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pth')
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'best_checkpoint.pth')
print(f"Model will save into: {CHECKPOINT_DIR}")
print(f"---------------------------\n")

start_epoch = 1
best_val_dice = None
not_improve_counter = None
if args.resume and os.path.exists(LAST_CHECKPOINT_PATH):
    try:
        if LOAD_BEST:
            checkpoint = torch.load(BEST_CHECKPOINT_PATH, map_location=DEVICE)
            print(f"\nResuming training from {BEST_CHECKPOINT_PATH}...")
        else:
            checkpoint = torch.load(LAST_CHECKPOINT_PATH, map_location=DEVICE)
            print(f"\nResuming training from {LAST_CHECKPOINT_PATH}...")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_dice = checkpoint.get('best_val_dice', None)
        not_improve_counter = checkpoint.get('not_improve_counter', None)

        if checkpoint.get('scheduler_state_dict', None) != None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint.get('is_early_stop', None):
            time.sleep(5)
            print("\nStop training by early stopping exit...\n")
            time.sleep(20)
            os._exit(0)
        
        if start_epoch >= NUM_EPOCHS:
            time.sleep(5)
            print(f"Stop training model is already trained at {NUM_EPOCHS} epoch")
            time.sleep(20)
            os._exit(0)

        print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting from scratch.")
        start_epoch = 1
        best_val_dice = None
        not_improve_counter = None

# ----------------------------------- MLFlow Configuration -----------------------------------
load_dotenv()
server_uri = os.getenv("MLFLOW_SERVER_URI")
mlflow.set_tracking_uri(server_uri)
mlflow.set_experiment(f"BrainTumor {model.model_name} Training")

# ----------------------------------- Training -----------------------------------
with mlflow.start_run(run_name=f"{model.model_name}_start-epoch{start_epoch}"):
    mlflow.set_tag("ml.step", "model_training_evaluation")
    mlflow.log_params(model.model_info)
    mlflow.log_param("start_epoch", start_epoch)
    mlflow.log_param("optimizer", OPTIMIZER_NAME)
    mlflow.log_param("target_epoch", NUM_EPOCHS)
    mlflow.log_param("train_data_dir", TRAIN_DATA_DIR)
    mlflow.log_param("val_data_dir", VAL_DATA_DIR)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("start_learning_rate", LR)
    mlflow.log_param("patience", PATIENCE)
    mlflow.log_param("se_reduction_rate", SE_REDUCTION)
    mlflow.log_param("dilation_rate", DILATION_RATE)
    mlflow.log_param("random_seed", SEED)
    mlflow.log_param("checkpoint_dir", CHECKPOINT_DIR)
    mlflow.log_param("crossentropy_loss_weight", LOSS_CE_WEIGHT)
    mlflow.log_param("dice_loss_weight", LOSS_DICE_WEIGHT)

    report = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion, 
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        start_epoch=start_epoch,
        device=DEVICE,
        num_classes=NUM_CLASSES,
        start_run_time=start_running_time,
        best_save_path=BEST_CHECKPOINT_PATH,
        last_save_path=LAST_CHECKPOINT_PATH,
        patience=PATIENCE,
        best_val_dice=best_val_dice,
        not_improve_counter=not_improve_counter
        )

if report is not None:
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y_%H-%M-%S")
    csv_file_name = f'{model.model_name}_{timestamp}.csv'
    report_path = os.path.join(TRAIN_RESULT_PATH, 'train_report_3', model.model_name)
    os.makedirs(report_path, exist_ok=True)
    report.to_csv(os.path.join(report_path, csv_file_name), index=False)

if 'model' in locals():
    del model
if 'optimizer' in locals():
    del optimizer

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

os._exit(0)