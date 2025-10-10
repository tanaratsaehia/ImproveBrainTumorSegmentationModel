import pandas as pd
import time
import torch

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

def train_model(model, criterion, optimizer, train_loader, val_loader, 
                num_epochs, num_classes, device):
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

    Returns:
        pandas.DataFrame: DataFrame containing epoch-wise metrics.
    """
    metrics_history = []
    
    print(f"Starting training on {device} for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # ------------------- TRAINING PHASE -------------------
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        num_batches_train = 0
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            
            # Remap label '4' -> '3' for 4 classes (0, 1, 2, 3)
            masks = masks.clone()
            masks[masks == 4] = 3

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            running_loss += loss.item()
            running_dice += dice_coef(outputs, masks, num_classes)
            running_iou += iou_metric(outputs, masks, num_classes)
            num_batches_train += 1

        train_loss = running_loss / num_batches_train
        train_dice = running_dice / num_batches_train
        train_iou = running_iou / num_batches_train

        # ------------------- VALIDATION PHASE -------------------
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
        
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch:02d}/{num_epochs} | Time: {epoch_time:.2f}s")
        print(f"  Train: Loss={train_loss:.4f}, Dice={train_dice:.4f}, IoU={train_iou:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Dice={val_dice:.4f}, IoU={val_iou:.4f}")
        
        # Store results
        metrics_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_dice': train_dice,
            'train_iou': train_iou,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'val_iou': val_iou,
            'time': epoch_time
        })
    return pd.DataFrame(metrics_history)