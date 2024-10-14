import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=0.25, reduction: str = 'mean'):

        super(FocalLoss, self).__init__()

        self.reduction = reduction
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):

        ce_loss = self.loss_fn(pred, target)

        p_t = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        return loss.mean() if self.reduction == 'mean' else \
              (loss.sum() if self.reduction == 'sum' else loss)

