# -*- coding: utf-8 -*-
"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Exploring Acoustic Emission Monitoring during Laser Powder Bed Fusion of premixed Ti6Al4V-Fe powder: Evidence of martensitic phase transformation supported by operando X-ray diffraction "
@any reuse of this code should be authorized by the first owner, code author

"""
# libraries to import
import torch.nn as nn
import torch
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise

    Args:
        margin (float): The margin value for the contrastive loss. Default is 1.0.

    Inputs:
        output1 (Tensor): Embeddings of the first sample. Shape (batch_size, embedding_size)
        output2 (Tensor): Embeddings of the second sample. Shape (batch_size, embedding_size)
        target (Tensor): Target labels indicating whether the samples are from the same class or not. Shape (batch_size,)

    Returns:
        Tensor: The contrastive loss value. Shape (1,)
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-6

    def forward(self, output1, output2, target):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean()
