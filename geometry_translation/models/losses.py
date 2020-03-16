import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()
        pass

    def forward(self, y_pred, y_true):
        smooth = 1.0  # may change
        i = torch.sum(y_true)
        j = torch.sum(y_pred)
        intersection = torch.sum(y_true * y_pred)
        score = (2. * intersection + smooth) / (i + j + smooth)
        loss = 1. - score.mean()
        return loss
