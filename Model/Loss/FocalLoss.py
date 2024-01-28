import torch.nn as nn
import torchvision


class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2., reduction = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, X, target):
        return torchvision.ops.sigmoid_focal_loss(X, target, self.alpha, self.gamma, self.reduction)

