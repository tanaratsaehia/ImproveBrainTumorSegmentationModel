import os
import pandas as pd
import time
import torch
from datetime import datetime
import mlflow
import mlflow.pytorch
from mlflow.artifacts import download_artifacts

def dice_coef(preds, targets, num_classes, smooth=1e-6):
    """
    Calculates the Dice Coefficient (F1 Score) for multi-class segmentation.
    Args:
        preds (Tensor): Model predictions (logits or probabilities) of shape (B, C, H, W).
        targets (Tensor): Ground truth masks of shape (B, H, W) with class indices.
    """
    # Convert predictions to class indices (B, H, W)
    preds = torch.argmax(preds, dim=1)
    
    dice_sum = 0.0
    
    for c in range(num_classes):
        # Create one-hot versions for the current class
        pred_c = (preds == c)
        target_c = (targets == c)
        
        # Flatten and compute intersection and union
        intersection = torch.sum(pred_c & target_c).item()
        union = torch.sum(pred_c).item() + torch.sum(target_c).item()
        
        # Dice formula
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_sum += dice
        
    # Mean Dice over all classes
    return dice_sum / num_classes

def iou_metric(preds, targets, num_classes, smooth=1e-6):
    """
    Calculates the Intersection over Union (Jaccard Index) for multi-class segmentation.
    """
    # Convert predictions to class indices (B, H, W)
    preds = torch.argmax(preds, dim=1)
    
    iou_sum = 0.0
    
    for c in range(num_classes):
        # Create one-hot versions for the current class
        pred_c = (preds == c)
        target_c = (targets == c)
        
        # Flatten and compute intersection and union
        intersection = torch.sum(pred_c & target_c).item()
        union = torch.sum(pred_c | target_c).item()
        
        # IoU formula
        iou = (intersection + smooth) / (union + smooth)
        iou_sum += iou
        
    # Mean IoU over all classes
    return iou_sum / num_classes

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, 
                num_epochs, num_classes, device, start_epoch, start_run_time, 
                best_save_path, last_save_path, patience=10, best_val_dice=None):
    """
    Trains and validates the segmentation model, tracking metrics per epoch.

    Args:
        model (nn.Module): The segmentation model (e.g., UNetDilationSE).
        criterion (nn.Module): Loss function (e.g., nn.CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): Optimizer (e.g., optim.Adam).
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        num_epochs (int): Total number of epochs to train for.
        num_classes (int): Number of segmentation classes (e.g., 4).
        device (torch.device): Device to run the model on ('cuda' or 'cpu').
        start_run_time (float): Start time to stop before hit gpu wall time.
        best_save_path (str): Path to save best model.
        last_save_path (str): Path to save last model.
        patience (int): For trigger early stopping.

    Returns:
        pandas.DataFrame: DataFrame containing epoch-wise metrics.
    """

    LIMIT_TIME_IN_SECONDS = (2 * 3600) + (50 * 60) # 2h 50m
    metrics_history = []
    print(f"Starting training on {device} for {num_epochs} epochs...")
    
    # Get total number of batches for the progress display
    total_batches = len(train_loader)
    best_val_dice = best_val_dice or float('-inf')
    epoch_not_improve_couter = 0
    
    
    for epoch in range(start_epoch, num_epochs+1):
        start_time = time.time()
        total_preprocess_time_each_epoch = 0
        total_model_training_time_each_epoch = 0
        
        # ------------------- TRAINING PHASE -------------------
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        num_batches_train = 0
        
        start_preprocess_time = time.time()
        for idx, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            total_preprocess_time_each_epoch += time.time() - start_preprocess_time

            # Remap label '4' -> '3' for 4 classes (0, 1, 2, 3)
            start_ai_compute_time = time.time()
            masks = masks.clone()
            masks[masks == 4] = 3

            optimizer.zero_grad()
            outputs = model(imgs)
            if isinstance(outputs, (list, tuple)):
                # Deep Supervision Case: outputs = [final, aux2, aux3]
                final_output = outputs[0]
                aux_out2 = outputs[1]
                aux_out3 = outputs[2]
                
                # Loss = Main + (0.5 * Aux2) + (0.5 * Aux3)
                # You can adjust the weight (0.5) as needed
                loss0 = criterion(final_output, masks)
                loss2 = criterion(aux_out2, masks)
                loss3 = criterion(aux_out3, masks)
                
                loss = loss0 + (0.5 * loss2) + (0.5 * loss3)
                
                # Use only final_output for metrics (Dice/IoU) logic below
                outputs = final_output 
            else:
                # Normal Case
                loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            current_loss = loss.item()
            running_loss += current_loss
            running_dice += dice_coef(outputs, masks, num_classes)
            running_iou += iou_metric(outputs, masks, num_classes)
            num_batches_train += 1

            if (idx + 1) % 100 == 0 or (idx + 1) == total_batches:
                # Calculate the average loss up to the current batch
                avg_running_loss = running_loss / num_batches_train
                print(
                    f"  [Epoch {epoch}/{num_epochs} | Batch {idx + 1}/{total_batches}] "
                    f"Batch Loss: {current_loss:.5f} | Avg. Train Loss: {avg_running_loss:.5f}"
                )
            
            if time.time() - start_run_time > LIMIT_TIME_IN_SECONDS:
                print(f"\nStop training before hit wall time at {epoch} epoch.\n")
                return pd.DataFrame(metrics_history)
            
            total_model_training_time_each_epoch += time.time() - start_ai_compute_time
            start_preprocess_time = time.time()


        train_loss = running_loss / num_batches_train
        train_dice = running_dice / num_batches_train
        train_iou = running_iou / num_batches_train

        # ------------------- VALIDATION PHASE -------------------
        start_val_time = time.time()
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        num_batches_val = 0
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                
                # Remap label '4' -> '3'
                masks = masks.clone()
                masks[masks == 4] = 3
                
                outputs = model(imgs)
                val_loss += criterion(outputs, masks).item()
                
                # Accumulate metrics
                val_dice += dice_coef(outputs, masks, num_classes)
                val_iou += iou_metric(outputs, masks, num_classes)
                num_batches_val += 1

        val_loss /= num_batches_val
        val_dice /= num_batches_val
        val_iou /= num_batches_val
        total_val_time = time.time() - start_val_time
        epoch_time = time.time() - start_time

        print("-" * 50)
        print(f"Epoch {epoch}/{num_epochs} Complete {epoch_time:.2f}s | Preprocess Time: {total_preprocess_time_each_epoch:.2f}s | Training time: {total_model_training_time_each_epoch:.2f}s | Validate time: {total_val_time:.2f}s")
        print(f"  Train Metrics: Loss={train_loss:.5f}, Dice={train_dice:.5f}, IoU={train_iou:.5f}")
        print(f"  Val Metrics:   Loss={val_loss:.5f}, Dice={val_dice:.5f}, IoU={val_iou:.5f}")
        
        if val_dice > best_val_dice:
            print(f"Validate Dice improved from {best_val_dice:.5f} to {val_dice:.5f}")
            print(f"Save new best model to '{best_save_path}'")
            best_val_dice = val_dice
            save_checkpoint(model, optimizer, scheduler, epoch, best_save_path, best_val_dice, val_loss)
            mlflow.log_artifact(best_save_path)
            mlflow.log_metric('best_saved_epoch', epoch)
            epoch_not_improve_couter = 0
        else:
            print(f"Model not impove dice...")
            epoch_not_improve_couter += 1
        
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")
        mlflow.log_metric("learning_rate", current_lr, step=epoch)
        print("-" * 50)
        
        # Store results
        epoch_result = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_dice': train_dice,
            'train_iou': train_iou,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'val_iou': val_iou,
            'epoch_time_sec': epoch_time
        }
        metrics_history.append(epoch_result)
        for key, value in epoch_result.items():
            if key != 'epoch':
                mlflow.log_metric(key, value, step=epoch)

        # Save last model
        save_checkpoint(model, optimizer, scheduler, epoch, last_save_path, best_val_dice, val_loss)
        # mlflow.log_artifact(last_save_path)                        # <<<<--------- Disable save last model into mlflow server for more training speed
        
        if epoch_not_improve_couter >= patience:
            print(f"Early stoping at {epoch} with {patience} patience")
            save_checkpoint(model, optimizer, scheduler, epoch, last_save_path, best_val_dice, val_loss, is_early_stop=True)
            return pd.DataFrame(metrics_history)
        
    print("\nFinish training!!\n")
    return pd.DataFrame(metrics_history)

def save_checkpoint(model, optimizer, scheduler, epoch, path, val_dice, val_loss=None, is_early_stop=False):
    """Saves the essential components needed to resume training."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_dice': val_dice,
        'val_loss': val_loss,
        'is_early_stop': is_early_stop,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(checkpoint, path)
    # print(f"\nCheckpoint saved to {path} (Completed Epoch: {epoch})")

def test_model(model, criterion, test_loader, num_classes, device):
    """
    Evaluates the segmentation model on a test/validation dataset.

    Args:
        model (nn.Module): The trained segmentation model.
        criterion (nn.Module): Loss function (e.g., HybridLoss).
        test_loader (DataLoader): DataLoader for the testing dataset.
        num_classes (int): Number of segmentation classes (e.g., 4).
        device (torch.device): Device to run the model on ('cuda' or 'cpu').

    Returns:
        pandas.DataFrame: DataFrame containing the final testing metrics.
    """
    print(f"--- Starting Final Testing on {device} ---")
    
    # Set the model to evaluation mode
    model.eval()
    
    test_loss = 0.0
    test_dice = 0.0
    test_iou = 0.0
    num_batches_test = 0
    start_time = time.time()
    
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            
            # Remap label '4' -> '3', consistent with training/validation
            masks = masks.clone()
            masks[masks == 4] = 3
            
            outputs = model(imgs)
            
            # Calculate loss and accumulate
            test_loss += criterion(outputs, masks).item()
            
            # Accumulate metrics
            test_dice += dice_coef(outputs, masks, num_classes)
            test_iou += iou_metric(outputs, masks, num_classes)
            num_batches_test += 1

    # Calculate final average metrics
    avg_test_loss = test_loss / num_batches_test
    avg_test_dice = test_dice / num_batches_test
    avg_test_iou = test_iou / num_batches_test
    
    test_time = time.time() - start_time

    print("-" * 50)
    print(f"Testing Complete | Total Time: {test_time:.2f}s")
    print(f"  Final Metrics: Loss={avg_test_loss:.5f}, Dice={avg_test_dice:.5f}, IoU={avg_test_iou:.5f}")
    print("-" * 50)
    
    # Store results in a Dict
    test_metrics = {
        'model_name': model.model_name,
        'loss': avg_test_loss,
        'dice': avg_test_dice,
        'iou': avg_test_iou,
        'time': test_time
    }
    
    return test_metrics