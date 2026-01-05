import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import pandas as pd # To handle the test report output

# --- Assume these are imported from their respective files ---
from data_helper import BRATSDataset2D, get_data_ids
from model_structure import *
from training_helper import test_model # The test function and combined loss
# Assume dice_coef and iou_metric are accessible if needed in this script, 
# but they are primarily used inside test_model.
# -----------------------------------------------------------

def main(args):
    # --- Configuration Setup ---
    NUM_CLASSES = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ROOT_DIR = os.path.join('BraTS-Datasets', args.data_dir)
    SLICE_INDICES = list(range(150))
    
    # 1. Instantiate the Model and Loss (CRITICAL: Must match training)
    # The structure must be created first before loading weights.
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
    # Use the same loss function used in training to report the test loss metric.
    criterion = HybridLoss(NUM_CLASSES) 
    
    # Define Checkpoint Path based on the model name
    CHECKPOINT_DIR = os.path.join('training_results', f'checkpoints_{model.model_name}')
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pth')
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
        return

    # 2. Load the Model Weights from Checkpoint
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    try:
        # Load the saved state dictionary
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        # Load the model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Determine the epoch the model was trained up to for naming the report
        trained_epoch = checkpoint.get('epoch', 'N/A') 
        print(f"Weights loaded successfully. Model trained up to epoch {trained_epoch}.")
    except Exception as e:
        print(f"Error loading model state: {e}")
        return

    # Move model to device and set to evaluation mode
    model.to(DEVICE)

    # 3. Data Loading (Load the Test or Validation Dataset)
    if not os.path.isdir(ROOT_DIR):
        print(f"Error: Directory not found at {ROOT_DIR}")
        return

    SUBJECT_IDS = get_data_ids(ROOT_DIR)
    
    # For a true test, ensure you partition the data correctly here.
    # We will use ALL data here, assuming it's the dedicated test set.
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

    # 4. Run the Test Function
    test_report_df = test_model(
        model=model,
        criterion=criterion,
        test_loader=test_loader,
        num_classes=NUM_CLASSES,
        device=DEVICE
    )

    # 5. Save the Test Report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f'FINAL_TEST_Ep{trained_epoch}_{model.model_name}_{timestamp}.csv'
    TEST_DIR = os.path.join('training_results', 'test_reports')
    os.makedirs(TEST_DIR, exist_ok=True)
    
    test_report_df.to_csv(os.path.join(TEST_DIR, report_name), index=False)
    print(f"\nFinal test report saved to {os.path.join(TEST_DIR, report_name)}")


if __name__ == '__main__':
    # Define only necessary arguments for testing
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