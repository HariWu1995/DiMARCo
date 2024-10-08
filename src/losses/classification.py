import torch
import torch.nn as nn


class NLLLoss(nn.Module):
    """
    Negative Log-Likelihood Loss (useful when combined with log-softmax)
    """
    def __init__(self):
        super(NLLLoss, self).__init__()
        self.loss_fn = nn.NLLLoss()

    def forward(self, predicted, target):
        return self.loss_fn(predicted, target)


class CrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, predicted, target):
        return self.loss_fn(predicted, target)


class BCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss (for binary classification)
    """
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, predicted, target):
        return self.loss_fn(predicted, target)


class BCEWithLogitsLoss(nn.Module):
    """
    Binary Cross-Entropy with Logits Loss (more numerically stable)
    """
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, predicted, target):
        return self.loss_fn(predicted, target)


class FocalLoss(BCEWithLogitsLoss):
    """
    Focal Loss (used for handling class imbalance)
    """
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__(reduction=None)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predicted, target):

        # Compute BCE with logits
        BCE_loss = super(FocalLoss, self).forward(predicted, target)

        # Convert predicted logits to probabilities
        pt = torch.sigmoid(predicted)

        # Compute focal loss
        pt = target * pt + (1 - target) * (1 - pt)  # probability of correct class
        focal_weight = (1 - pt) ** self.gamma       # Focusing factor        
        focal_loss = self.alpha * focal_weight * BCE_loss

        return torch.mean(focal_loss)

