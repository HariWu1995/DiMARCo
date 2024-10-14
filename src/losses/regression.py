import torch
import torch.nn as nn

from ..const import eps


class MSELoss(nn.Module):
    """
    Mean Squared Error Loss
    """
    def __init__(self, reduction: str = 'mean'):
        super(MSELoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, predicted, target):
        return self.loss_fn(predicted, target)


class MAELoss(nn.Module):
    """
    Mean Absolute Error Loss (L1 Loss)
    """
    def __init__(self, reduction: str = 'mean'):
        super(L1Loss, self).__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)

    def forward(self, predicted, target):
        return self.loss_fn(predicted, target)


class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1 Loss)
    """
    def __init__(self, reduction: str = 'mean', beta: float = 1.0):
        super(HuberLoss, self).__init__()
        self.loss_fn = nn.SmoothL1Loss(reduction=reduction, beta=beta)

    def forward(self, predicted, target):
        return self.loss_fn(predicted, target)


class LogCoshLoss(nn.Module):
    """
    Log-Cosh Loss
    """
    def __init__(self, reduction: str = 'mean'):
        super(LogCoshLoss, self).__init__()
        self.reduction = reduction

    def forward(self, predicted, target):
        diff = predicted - target
        loss = torch.log(torch.cosh(diff))
        return loss.mean() if self.reduction == 'mean' else \
              (loss.sum() if self.reduction == 'sum' else loss)


class MAPELoss(nn.Module):
    """
    Mean Absolute Percentage Error (MAPE)
    """
    def __init__(self, reduction: str = 'mean'):
        super(MAPELoss, self).__init__()
        self.reduction = reduction

    def forward(self, predicted, target):
        loss = torch.abs((target - predicted) / (target + eps)) * 100.
        return loss.mean() if self.reduction == 'mean' else \
              (loss.sum() if self.reduction == 'sum' else loss)


class RSquaredLoss(nn.Module):
    """
    R-squared (Coefficient of Determination)
    """
    def __init__(self):
        super(RSquaredLoss, self).__init__()

    def forward(self, predicted, target):
        ss_res = torch.sum((target - predicted) ** 2)
        ss_tot = torch.sum((target - torch.mean(target)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + eps)
        return r_squared


class AdjustedRSquaredLoss(nn.Module):
    """
    Adjusted R-squared (Adjusted Coefficient of Determination)
    
    Parameters:
    - num_features (int): The number of features (predictors) in the model.
    """
    def __init__(self, num_features):
        super(AdjustedRSquaredLoss, self).__init__()
        self.num_features = num_features

    def forward(self, predicted, target):
        n = target.size(0)      # Number of samples
        p = self.num_features   # Number of features (predictors)

        ss_res = torch.sum((target - predicted) ** 2)
        ss_tot = torch.sum((target - torch.mean(target)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + eps)

        # Adjusted R-squared formula
        adj_r_squared = 1 - ((1 - r_squared) * (n - 1)) / (n - p - 1 + eps)
        return adj_r_squared


class BoundaryLoss(nn.Module):

    def __init__(self, reduction: str = 'mean'):
        super(BoundaryLoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target, mask):
        boundary_pred   = torch.abs(torch.gradient(  pred, dim=1)) * mask
        boundary_target = torch.abs(torch.gradient(target, dim=1)) * mask
        
        return F.mse_loss(boundary_pred, boundary_target, reduction=self.reduction)

