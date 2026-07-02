import torch
import torch.nn as nn


class EPE(nn.Module):
    def forward(self, flow, gt, loss_mask=None):
        loss = ((flow - gt) ** 2).sum(1).sqrt()
        if loss_mask is not None:
            loss = loss * loss_mask
        return loss.mean()


class SOBEL(nn.Module):
    def forward(self, pred, gt):
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
