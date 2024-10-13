import torch
from torch import nn
from torch.nn import functional as F

from ..const import eps


class WeightedLayeredLoss(nn.Module):

    def __init__( self, foreground_loss_fn, 
                        background_loss_fn, 
                        foreground_weight: float = 0.9, 
                        background_weight: float = None):
        """
        Combines foreground and background losses with different weights.
        
        Args:
            foreground_loss_fn: Loss function for the foreground (e.g., BCE, MSE).
            background_loss_fn: Loss function for the background.
            foreground_weight: Weight for the foreground loss.
            background_weight: Weight for the background loss.
        """
        super(WeightedCombinedLoss, self).__init__()
        self.fg_loss_fn = foreground_loss_fn
        self.bg_loss_fn = background_loss_fn
        self.fg_weight = foreground_weight
        self.bg_weight = background_weight if isinstance(background_weight, (int, float)) else \
                        (1-foreground_weight)

    def forward(self, pred, target, mask):
        """
        Args:
            pred: Predictions from the model.
            target: Ground truth labels.
            mask: Binary mask (1 for foreground, 0 for background).
        """
        fg_mask = mask
        bg_mask = 1 - mask

        fg_loss = self.fg_loss_fn(pred * fg_mask, target * fg_mask)
        bg_loss = self.bg_loss_fn(pred * bg_mask, target * bg_mask)

        loss = self.fg_weight * fg_loss \
             + self.bg_weight * bg_loss

        return loss.mean()


class LayeredFocalLoss(WeightedLayeredLoss):

    def __init__( self, gamma = 2, alpha = 0.25, beta = 0.5, 
                        foreground_weight: float = 0.9, 
                        background_weight: float = None):

        from .classification import FocalLoss
        from .regression import MSELoss

        super(LayeredFocalLoss, self).__init__(
                foreground_loss_fn = FocalLoss(gamma=gamma, alpha=alpha, reduction='none'),
                background_loss_fn = MSELoss(),
                foreground_weight = foreground_weight,
                background_weight = background_weight,                
            )


class BoundaryAwareLoss(nn.Module):

    def __init__( self, foreground_weight: float = 0.9, 
                        background_weight: float = None):
        """
        Boundary-aware loss to emphasize errors at object boundaries.
        """
        from .regression import MSELoss, BoundaryLoss

        super(BoundaryAwareLoss, self).__init__(
                foreground_loss_fn = MSELoss(),
                background_loss_fn = BoundaryLoss(),
                foreground_weight = foreground_weight,
                background_weight = background_weight,                
            )



