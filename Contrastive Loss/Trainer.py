# -*- coding: utf-8 -*-
"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in Laser Powder Bed Fusion Process"
@any reuse of this code should be authorized by the first owner, code author

"""
# libraries to import
from tqdm.notebook import tqdm
import torch
import numpy as np
import os


def model_trainer(epochs, train_loader, device, optimizer, model, criterion, Exptype, folder_created):
    running_loss = []
    Training_loss = []

    Training_loss_mean = []
    Training_loss_std = []

    for epoch in tqdm(range(epochs), desc="Epochs"):
        total_loss = 0
        iterations = 0
        epoch_smoothing = []

        for step, (anchor_img, pair_img, anchor_label, label) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            #torch.Size([100, 1, 5000])
            anchor_img = anchor_img.to(device, dtype=torch.float)
            anchor_img = anchor_img.unsqueeze(1)
            # print(anchor_label)

            anchor_label = anchor_label.to(device, dtype=torch.float)

            pair_img = pair_img.to(device, dtype=torch.float)
            pair_img = pair_img.unsqueeze(1)
            # print(positive_img.shape)

            optimizer.zero_grad()
            anchor_out = model(anchor_img)
            pair_out = model(pair_img)
            # negative_out = model(negative_img)

            loss = criterion(anchor_out, pair_out, anchor_label)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy())
            epoch_smoothing.append(loss.cpu().detach().numpy())

            total_loss += loss

            iterations = iterations+1

        loss_train = total_loss/len(train_loader)
        Training_loss.append(loss_train.cpu().detach().numpy())
        Training_loss_mean.append(np.mean(epoch_smoothing))
        Training_loss_std.append(np.std(epoch_smoothing))

        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, loss_train))

    model_name = Exptype+"_trained_model.pth"
    torch.save({"model_state_dict": model.state_dict(),
                "optimzier_state_dict": optimizer.state_dict()
                },  os.path.join(folder_created, model_name))

    return Training_loss, Training_loss_mean, Training_loss_std, running_loss, model


def save_train_embeddings(device, Exptype, folder_created, train_loader, model):
    train_results = []
    labels = []

    model.eval()
    with torch.no_grad():
        for img, _, _, label in tqdm(train_loader):
            # print(img.shape)
            img = img.unsqueeze(1)
            # print(img.shape)
        # img=img.unsqueeze(1)
            train_results.append(model(img.to(device, dtype=torch.float)).cpu().numpy())
            labels.append(label)

    train_results = np.concatenate(train_results)
    train_labels = np.concatenate(labels)
    train_results.shape

    train_embeddings = Exptype+'_train_embeddings_.npy'
    train_embeddings = os.path.join(folder_created, train_embeddings)
    train_labelsname = Exptype+'_train_labels_.npy'
    train_labelsname = os.path.join(folder_created, train_labelsname)

    np.save(train_embeddings, train_results, allow_pickle=True)
    np.save(train_labelsname, train_labels, allow_pickle=True)
    return train_results, train_labels


def save_test_embeddings(device, Exptype, folder_created, test_loader, model):
    test_results = []
    labels = []

    model.eval()
    with torch.no_grad():
        for img, label in tqdm(test_loader):

            img = img.unsqueeze(1)

            test_results.append(model(img.to(device, dtype=torch.float)).cpu().numpy())
            labels.append(label)

    test_results = np.concatenate(test_results)
    test_labels = np.concatenate(labels)
    test_results.shape

    test_embeddings = Exptype+'_test_embeddings_.npy'
    test_embeddings = os.path.join(folder_created, test_embeddings)

    test_labelsname = Exptype+'_test_labels_.npy'
    test_labelsname = os.path.join(folder_created, test_labelsname)

    np.save(test_embeddings, test_results, allow_pickle=True)
    np.save(test_labelsname, test_labels, allow_pickle=True)
    return test_results, test_labels
