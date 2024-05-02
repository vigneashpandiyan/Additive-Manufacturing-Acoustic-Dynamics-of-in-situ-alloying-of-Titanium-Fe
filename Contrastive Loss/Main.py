# -*- coding: utf-8 -*-
"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in Laser Powder Bed Fusion Process"
@any reuse of this code should be authorized by the first owner, code author

"""
# libraries to import

import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from Utils import *
from Network import *
from Dataloader import *
from Loss import *
from Classifier import *
from Trainer import *
from matplotlib import animation
import os

# %%
# Clearing the cache
torch.cuda.empty_cache()
torch.manual_seed(2020)
np.random.seed(2020)
random.seed(2020)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.get_device_name()
print('Using device:', device)

# %%
datapath = r'C:\Users\srpv\Desktop\Git\Additive-Manufacturing-Acoustic-Dynamics-of-in-situ-alloying-of-Titanium-Fe\ML classifier\Data_preprocessing'
embedding_dims = 8
batch_size = 64
epochs = 300

Datasets = ['CM', 'KM']

for Exptype in Datasets:

    category = ['Ti64', 'Ti64-3Fe', 'Ti64-6Fe']
    class_names = []
    for x in category:
        x = str(Exptype)+'-'+x
        class_names.append(x)

    # %%
    # folder creation to store latent spaces, graphs and figure.
    file = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
    total_path = os.path.dirname(file)
    print(total_path)

    folder_created = os.path.join(total_path, Exptype)
    print(folder_created)
    try:
        os.makedirs(folder_created, exist_ok=True)
        print("Directory created....")
    except OSError as error:
        print("Directory already exists....")

    # %%
    data_plot(datapath, Exptype, folder_created)
    train_df, test_df = data_extract(datapath, Exptype)
    train_loader, test_loader = torch_loader(batch_size, train_df, test_df)

    # %%

    model = Network(embedding_dims)
    model.apply(init_weights)
    model = torch.jit.script(model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.jit.script(ContrastiveLoss())
    model.train()
    Training_loss, Training_loss_mean, Training_loss_std, running_loss, model = model_trainer(
        epochs, train_loader, device, optimizer, model, criterion, Exptype, folder_created)

    # %%

    plot_function(Exptype, folder_created, Training_loss,
                  Training_loss_mean, Training_loss_std, running_loss)

    # %%

    train_results, train_labels = save_train_embeddings(
        device, Exptype, folder_created, train_loader, model)

    test_results, test_labels = save_test_embeddings(
        device, Exptype, folder_created, test_loader, model)

    # %%

    graph_name_2D = os.path.join(folder_created, 'Training_Feature_2D.png')
    plot_embeddings(train_results, train_labels, graph_name_2D, class_names)

    graph_name_2D = os.path.join(folder_created, 'Testing_Feature_2D.png')
    plot_embeddings(test_results, test_labels, graph_name_2D, class_names)

    # %%

    latent_animation(class_names, folder_created, train_results,
                     train_labels, test_results, test_labels)

    # %%
    count_parameters(model)

    # %%
    classifier_linear(Exptype, folder_created)
