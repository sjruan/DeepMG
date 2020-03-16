import torch
import torch.nn as nn


class Metrics(nn.Module):
    def __init__(self, threshold=0.5):
        super(Metrics, self).__init__()
        self.threshold = threshold

    def forward(self, pred, true):
        eps = 1e-10
        pred_ = (pred > self.threshold).data.float()
        true_ = (true > self.threshold).data.float()
        intersection = torch.clamp(pred_ * true_, 0, 1)
        union = torch.clamp(pred_ + true_, 0, 1)
        if torch.mean(intersection).lt(eps):
            return torch.tensor([0., 0., 0., 0.])
        else:
            iou = torch.mean(intersection) / torch.mean(union)
            precision = torch.mean(intersection) / torch.mean(pred_)
            recall = torch.mean(intersection) / torch.mean(true_)
            f1 = 2 * precision * recall / (precision + recall)
            return torch.tensor([precision, recall, f1, iou])
