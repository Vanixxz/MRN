import torch
import torch.nn as nn
import torch.nn.functional as F

class Softmax(nn.Module):
    def __init__(self, **options):
        super(Softmax, self).__init__()

    def forward(self, x, y, labels=None):
        logits = F.softmax(y, dim=1)
        if labels is None: return logits, 0
        loss = F.cross_entropy(y, labels)
        return logits, loss