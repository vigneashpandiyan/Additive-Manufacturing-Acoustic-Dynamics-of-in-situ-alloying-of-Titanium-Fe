# -*- coding: utf-8 -*-
"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Exploring Acoustic Emission Monitoring during Laser Powder Bed Fusion of premixed Ti6Al4V-Fe powder: Evidence of martensitic phase transformation supported by operando X-ray diffraction "
@any reuse of this code should be authorized by the first owner, code author

"""
# libraries to import
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import torch.nn as nn
import torch


class Network(nn.Module):
    """
    This class represents a neural network model for processing 1D input data.

    Args:
        emb_dim (int): The dimension of the output embedding. Default is 4.

    Attributes:
        conv (nn.Sequential): The convolutional layers of the network.
        fc (nn.Sequential): The fully connected layers of the network.
    """

    def __init__(self, emb_dim=4):
        super(Network, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=8, stride=8),
            nn.BatchNorm1d(4),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(4, 8, kernel_size=8, stride=8),
            nn.BatchNorm1d(8),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(8, 16, kernel_size=8, stride=4),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(16, 32, kernel_size=8, stride=4),
            nn.BatchNorm1d(32),
            nn.PReLU(),
            nn.Dropout(0.1),
        )

        self.fc = nn.Sequential(
            nn.Linear(32*3, 64),
            nn.PReLU(),
            nn.Linear(64, emb_dim),
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 1, sequence_length).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, emb_dim).
        """
        x = self.conv(x)
        x = x.view(-1, 32*3)
        x = self.fc(x)
        return x
