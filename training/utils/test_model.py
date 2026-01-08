import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import pandas as pd 

from data_helper import BRATSDataset2D, get_data_ids
from model_structure import *
from training_helper import test_model 

def main(args):
    NUM_CLASSES = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ROOT_DIR = os.path.join('BraTS-Datasets', args.data_dir)
    SLICE_INDICES = list(range(150))
    
    model = None
    if args.model_name == 'u_net': #Model name (u_net, u_net_se, u_net_pyramid, u_net_pyramid_se)
        model = UNet(in_channels=4, num_classes=NUM_CLASSES)
    elif args.model_name == 'u_net_se':
        model = UNetSE(in_channels=4, num_classes=NUM_CLASSES)
    elif args.model_name == 'u_net_pyramid':
        model = UNetBiPyramid(in_channels=4, num_classes=NUM_CLASSES)
    elif args.model_name == 'u_net_pyramid_se':
        model = UNetBiPyramidSE(in_channels=4, num_classes=NUM_CLASSES)
    elif args.model_name == 'u_net_pyramid_di':
        model = UNetBiPyramidDi(in_channels=4, num_classes=NUM_CLASSES)
    else:
        raise("model name not correct")
    print(f'Testing model {model.model_name}')
    criterion = HybridLoss(NUM_CLASSES) 
    
    CHECKPOINT_DIR = os.path.join('training_results', f'checkpoints_{model.model_name}')
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pth')
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
        return

    print(f"Loading weights from {CHECKPOINT_PATH}...")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        trained_epoch = checkpoint.get('epoch', 'N/A') 
        print(f"Weights loaded successfully. Model trained up to epoch {trained_epoch}.")
    except Exception as e:
        print(f"Error loading model state: {e}")
        return

    model.to(DEVICE)
    if not os.path.isdir(ROOT_DIR):
        print(f"Error: Directory not found at {ROOT_DIR}")
        return

    SUBJECT_IDS = get_data_ids(ROOT_DIR)
    test_dataset = BRATSDataset2D(
        root_dir=ROOT_DIR,
        subject_ids=SUBJECT_IDS,
        slice_indices=SLICE_INDICES,
        transform=None
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=2, 
        pin_memory=True
    )
    print(f'Total test data: {len(test_dataset)} images.')

    test_report_df = test_model(
        model=model,
        criterion=criterion,
        test_loader=test_loader,
        num_classes=NUM_CLASSES,
        device=DEVICE
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f'FINAL_TEST_Ep{trained_epoch}_{model.model_name}_{timestamp}.csv'
    TEST_DIR = os.path.join('training_results', 'test_reports')
    os.makedirs(TEST_DIR, exist_ok=True)
    
    test_report_df.to_csv(os.path.join(TEST_DIR, report_name), index=False)
    print(f"\nFinal test report saved to {os.path.join(TEST_DIR, report_name)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run final model evaluation.")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='BraTS-10file-testset',
        help="Path to the root directory containing the BraTS test/val datasets."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help="Batch size for DataLoader (default: 16)"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='u_net',
        help="Model name (u_net, u_net_se, u_net_pyramid, u_net_pyramid_se)"
    )
    args = parser.parse_args()
    from datetime import datetime
    main(args)