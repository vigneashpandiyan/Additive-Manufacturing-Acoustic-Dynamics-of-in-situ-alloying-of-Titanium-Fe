# -*- coding: utf-8 -*-
"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in Laser Powder Bed Fusion Process"
@any reuse of this code should be authorized by the first owner, code author

"""
# libraries to import
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import torch.nn as nn
import torch


class Network(nn.Module):
    def __init__(self, emb_dim=4):
        super(Network, self).__init__()
        #torch.Size([100, 1, 5000])

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
        x = self.conv(x)

        # print(x.shape)
        x = x.view(-1, 32*3)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x
