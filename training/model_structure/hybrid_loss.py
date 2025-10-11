import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss for PyTorch.
    Assumes inputs are logits/probabilities (N, C, H, W) and targets are class indices (N, H, W).
    """
    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Convert inputs (logits) to probabilities
        inputs = F.softmax(inputs, dim=1)  
        
        # One-hot encode the targets (N, H, W) -> (N, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1)
        targets_one_hot = targets_one_hot.contiguous().view(-1)
        
        # Calculate intersection and union
        intersection = (inputs * targets_one_hot).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets_one_hot.sum() + self.smooth)
        
        # Return 1 - Dice coefficient
        return 1 - dice

class HybridLoss(nn.Module):
    """
    Combines Cross-Entropy Loss and Dice Loss with weights.
    """
    def __init__(self, num_classes, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
    def forward(self, inputs, targets):
        # The Cross-Entropy loss expects inputs (logits) and targets (class indices)
        ce_loss = self.ce_loss(inputs, targets)
        
        # The Dice Loss also expects inputs (logits) and targets (class indices)
        dice_loss = self.dice_loss(inputs, targets)
        
        # Combine the losses
        total_loss = (self.ce_weight * ce_loss) + (self.dice_weight * dice_loss)
        return total_loss