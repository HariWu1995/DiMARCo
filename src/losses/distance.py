import torch
from torch import nn
from torch.nn import functional as F


class SquaredGreatCircleDistanceLoss(nn.Module):

    def __init__(self):
        super(SquaredGreatCircleDistanceLoss, self).__init__()

    def forward(self, u, v):
        """
        Compute the Squared Great Circle Distance Loss between 2 tensors u and v.

        Args:
            - u: Tensor of shape (batch_size, num_features)
            - v: Tensor of shape (batch_size, num_features)

        Returns:
            - loss: Squared great circle distance between u and v.
        """
        ## Ensure vectors are unit vectors (on the surface of a sphere)
        u_norm = u / u.norm(dim=-1, keepdim=True)
        v_norm = v / v.norm(dim=-1, keepdim=True)

        ## Squared Great Circle Distance
        ## Reference: ChatGPT
        # dot_product = torch.sum(u_norm * v_norm, dim=-1)
        # dot_product = torch.clamp(dot_product, -1., 1.)
        # distance = torch.acos(dot_product) ** 2

        ## Squared Great Circle Distance
        ## Reference: 
        ##    `clip_loss` in https://huggingface.co/learn/diffusion-course/unit2/2
        distance = (u_norm - v_norm).norm(dim=-1)
        distance = distance.div(2).arcsin().pow(2).mul(2)

        return distance.mean()

