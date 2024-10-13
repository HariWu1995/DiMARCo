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

    def __init__(self, gamma=0, alpha=None, reduction: str = 'mean'):

        super(FocalLoss, self).__init__()

        self.reduction = reduction
        self.gamma = gamma

        if isinstance(alpha, (float, int)): 
            self.alpha = torch.Tensor([alpha, 1-alpha])
        elif isinstance(alpha, (list, tuple)): 
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = None

    def forward(self, pred, target):
        if pred.dim() > 2:
            pred = pred.view(pred.size(0), pred.size(1), -1)    # N, C, H,W => N, C, H*W
            pred = pred.transpose(1,2)                          # N, C, H*W => N, H*W, C
            pred = pred.contiguous().view(-1, pred.size(2))     # N, H*W, C => N*H*W, C

        target = target.view(-1,1)

        logpt = F.log_softmax(pred)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != pred.data.type():
                self.alpha = self.alpha.type_as(pred.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.reduction == 'mean': 
            return loss.mean()
        elif self.reduction == 'sum': 
            return loss.sum()
        else:
            return loss

