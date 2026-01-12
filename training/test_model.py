import os
import torch
import random
import mlflow
import argparse
import numpy as np
import pandas as pd 
import torch.nn as nn
from datetime import datetime
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from model_structure import *
from utils.training_helper import test_model 
from utils.data_helper import BRATSDataset2D

def main(args):
    NUM_CLASSES = 4
    DATA_DIR = args.data_dir
    MODEL_NAME = args.model_name
    SE_REDUCTION = args.se_reduction
    DILATION_RATE = args.dilation_rate
    CHECKPOINT_PATH = args.weight_path
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    elif MODEL_NAME == "u_net_ag":
        model = UNetAG(in_channels=4, num_classes=NUM_CLASSES)
    elif MODEL_NAME == "u_net_aspp":
        model = UNetASPP(in_channels=4, num_classes=NUM_CLASSES)
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
    print(f"\nModel info: {model.model_info}")
    criterion = HybridLoss(NUM_CLASSES, ce_weight=0.5, dice_weight=0.5) 
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
        return

    print(f"Loading weights from {CHECKPOINT_PATH}")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        trained_epoch = checkpoint.get('epoch', 'N/A') 
        print(f"Weights loaded successfully. Model trained from epoch {trained_epoch}.")
    except Exception as e:
        print(f"Error loading model state: {e}")
        return

    with mlflow.start_run(run_name=f"TEST_{model.model_name}"):
        mlflow.set_tag("ml.step", "model_testing")
        mlflow.log_param("model_info", model.model_info)
        mlflow.log_param("weight_from_epoch", trained_epoch)
        mlflow.log_param("data_dir", DATA_DIR)

        model.to(DEVICE)
        for data_path in DATA_DIR:
            test_prefix = 'unknow_test_data'
            if 'low_grade' in data_path:
                test_prefix = 'low_grade'
            elif 'high_grade' in data_path:
                test_prefix = 'high_grade'
            
            if not os.path.isdir(data_path):
                print(f"Error: Directory not found at {data_path}")
                return

            test_dataset = BRATSDataset2D(
                csv_path    = os.path.join(data_path, 'dataset_mapper.csv'),
                root_dir    = data_path,
                transform   = None
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=args.batch_size, 
                shuffle=False,
                num_workers=2, 
                pin_memory=False
            )
            print(f'Total test data: {len(test_dataset)} images.')

            test_report_dict = test_model(
                model=model,
                criterion=criterion,
                test_loader=test_loader,
                num_classes=NUM_CLASSES,
                device=DEVICE
            )
            
            for key, value in test_report_dict.items():
                if key not in ['model_name', 'time']:
                    mlflow.log_metric(f"{test_prefix}_{key}", value)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f'TEST_{test_prefix}_epoch{trained_epoch}_{model.model_name}_{timestamp}.csv'
            TEST_DIR = os.path.join('training_results', 'test_reports')
            os.makedirs(TEST_DIR, exist_ok=True)
            
            test_report_df = pd.DataFrame([test_report_dict])
            test_report_df.to_csv(os.path.join(TEST_DIR, report_name), index=False)
            print(f"Final test report saved to {os.path.join(TEST_DIR, report_name)}\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run final model evaluation.")
    parser.add_argument(
        'model_name',
        type=str,
        choices=["u_net", "u_net_se", "u_net_di", "u_net_se_di", "u_net_ag", "u_net_aspp", "bipyramid", "bipyramid_se", "bipyramid_di", "bipyramid_se_di"],
        help="Name of the architecture to use. Options: %(choices)s"
    )
    parser.add_argument(
        'weight_path',
        type=str
    )
    parser.add_argument(
        '--data_dir',
        type=list,
        default=['BraTS-Datasets/Testset/SLICED_low_grade', 'BraTS-Datasets/Testset/SLICED_high_grade'],
        help="Path to the root directory containing high and low grade datasets."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help="Batch size for DataLoader (default: 16)"
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
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    args = parser.parse_args()
    
    load_dotenv()
    server_uri = os.getenv("MLFLOW_SERVER_URI")
    mlflow.set_tracking_uri(server_uri)
    mlflow.set_experiment(f"Testing models")

    print()
    main(args)

# u_net training_results/checkpoints_U-Net/best_checkpoint.pth
# u_net_se training_results/checkpoints_U-Net_SE/best_checkpoint.pth
# u_net_di training_results/checkpoints_U-Net_DI1212/best_checkpoint.pth
# u_net_se_di training_results/checkpoints_U-Net_SE_DI1212/best_checkpoint.pth
# u_net_ag training_results/checkpoints_U-Net_AG/best_checkpoint.pth
# u_net_aspp training_results/checkpoints_U-Net_ASPP/best_checkpoint.pth

# bipyramid training_results/checkpoints_U-Net_BiPyramid/best_checkpoint.pth
# bipyramid_se training_results/checkpoints_U-Net_BiPyramid_SE/best_checkpoint.pth
# bipyramid_di training_results/checkpoints_U-Net_BiPyramid_DI1212/best_checkpoint.pth
# bipyramid_se_di training_results/checkpoints_U-Net_BiPyramid_SE_DI1212/best_checkpoint.pth
