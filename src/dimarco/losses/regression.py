import torch
import torch.nn as nn


eps = 1e-7


class MSELoss(nn.Module):
    """
    Mean Squared Error Loss
    """
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, predicted, target):
        return self.loss_fn(predicted, target)


class MAELoss(nn.Module):
    """
    Mean Absolute Error Loss (L1 Loss)
    """
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, predicted, target):
        return self.loss_fn(predicted, target)


class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1 Loss)
    """
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, predicted, target):
        return self.loss_fn(predicted, target)


class LogCoshLoss(nn.Module):
    """
    Log-Cosh Loss
    """
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, predicted, target):
        diff = predicted - target
        return torch.mean(torch.log(torch.cosh(diff)))


class MAPELoss(nn.Module):
    """
    Mean Absolute Percentage Error (MAPE)
    """
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, predicted, target):
        percentage_error = torch.abs((target - predicted) / (target + eps))
        return torch.mean(percentage_error) * 100


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

