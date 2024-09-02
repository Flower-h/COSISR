import torch
from torch import nn as nn
from torch.nn import functional as F
import math
import numpy as np

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.loss_util import weighted_loss
from basicsr.losses.basic_loss import l1_loss, L1Loss

_reduction_modes = ['none', 'mean', 'sum']


@LOSS_REGISTRY.register()
class WSL1Loss(nn.Module):
    """WS-L1 loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, h=480, reduction='mean'):
        super(WSL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.h = h

    def forward(self, pred, target, top, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
            top: (Tensor): of shape (N, H). Start latitude of patches.
        """
        top = top[0] * torch.ones(pred.shape[0], pred.shape[2])
        a = 0.4308
        b = 0.2303
        c = 0.4885
        dx = 0.00427
        m0 = 170
        f = 5.
        L = 1.1529 * torch.ones(pred.shape[2])
        H = a ** 2 + c ** 2
        J = 2 * a * c
        _weight = top + torch.tensor(np.linspace(0, pred.shape[2] - 1, pred.shape[2])).unsqueeze(
            0).repeat(pred.shape[0], 1)
        y = (m0 * torch.ones(pred.shape[2]) - _weight) * dx
        P = torch.sqrt(L ** 2 + y ** 2)
        _weight = f ** 2 * b ** 4 * L * (H * P - J * y) / ((J * P - H * y) ** 3 * P)
        _weight = _weight.unsqueeze(1).unsqueeze(-1)
        _weight = _weight.repeat(1, pred.shape[1], 1, pred.shape[3]).detach().cuda(pred.device)
        return self.loss_weight * l1_loss(pred, target, weight=_weight, reduction=self.reduction)
