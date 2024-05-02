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
import pandas as pd
import os
from matplotlib import animation


def plot_function(Exptype, folder_created, Training_loss, Training_loss_mean, Training_loss_std, running_loss):
    Training_loss = np.asarray(Training_loss)
    Training_lossfile = Exptype+'_Training_loss.npy'
    Training_lossfile = os.path.join(folder_created, Training_lossfile)
    np.save(Training_lossfile, Training_loss, allow_pickle=True)

    Training_loss_mean = np.asarray(Training_loss_mean)
    Training_loss_meanfile = Exptype+'_Training_loss_mean.npy'
    Training_loss_meanfile = os.path.join(folder_created, Training_loss_meanfile)
    np.save(Training_loss_meanfile, Training_loss_mean, allow_pickle=True)

    Training_loss_std = np.asarray(Training_loss_std)
    Training_loss_stdfile = Exptype+'_Training_loss_std.npy'
    Training_loss_stdfile = os.path.join(folder_created, Training_loss_stdfile)
    np.save(Training_loss_stdfile, Training_loss_std, allow_pickle=True)

    plt.rcParams.update({'font.size': 15})
    plt.figure(1)
    plt.plot(running_loss, 'b--', linewidth=2.0)
    plt.title('Iteration vs Loss value')
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')
    plt.savefig(os.path.join(folder_created, 'Training loss.png'), dpi=600, bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots()
    plt.plot(Training_loss, 'g', linewidth=1.0)
    ax.fill_between(Training_loss, Training_loss_mean - Training_loss_std,
                    Training_loss_mean + Training_loss_std, alpha=0.5)
    ax.legend(['Training loss'])
    plt.title('Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.savefig(os.path.join(folder_created, 'Training loss average.png'),
                dpi=600, bbox_inches='tight')
    plt.show()


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


marker = ["o", "s", "d", "*", ">", "X"]
color = ['cyan', 'orange', 'purple', 'blue', 'green', 'red']


def plot_embeddings(embeddings, targets, graph_name_2D, classes, xlim=None, ylim=None):
    plt.figure(figsize=(7, 5))
    count = 0
    for i in np.unique(targets):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.7,
                    color=color[count], marker=marker[count], s=100)
        count = count+1
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes, bbox_to_anchor=(1.41, 1.05))
    plt.xlabel('Weights_1', labelpad=10)
    plt.ylabel('Weights_2', labelpad=10)
    graph_title = "Feature space distribution"
    plt.title(str(graph_title), fontsize=15)
    plt.savefig(graph_name_2D, bbox_inches='tight', dpi=600)
    plt.show()


def Three_embeddings(embeddings, targets, graph_name, class_names, ang, xlim=None, ylim=None):
    group = targets

    df2 = pd.DataFrame(group)
    df2.columns = ['Categorical']

    df2 = df2['Categorical'].replace(0, 'Ti64')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(1, 'Ti64_3Fe')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(2, 'Ti64_6Fe')
    df2 = pd.DataFrame(df2)

    group = pd.DataFrame(df2)

    group = group.to_numpy()
    group = np.ravel(group)

    x1 = embeddings[:, 0]
    x2 = embeddings[:, 1]
    x3 = embeddings[:, 2]

    df = pd.DataFrame(dict(x=x1, y=x2, z=x3, label=group))
    groups = df.groupby('label')
    uniq = list(set(df['label']))
    uniq = ['Ti64', 'Ti64_3Fe', 'Ti64_6Fe']
    # uniq = class_names
    # uniq=["0","1","2","3"]

    fig = plt.figure(figsize=(12, 6), dpi=100)
    fig.set_facecolor('white')
    plt.rcParams["legend.markerscale"] = 2

    ax = plt.axes(projection='3d')

    ax.grid(False)
    ax.view_init(azim=ang)  # 115
    marker = ["o", "s", "d", "*", ">", "X"]
    color = ['cyan', 'orange', 'purple', 'blue', 'green', 'red']

    ax.set_facecolor('white')
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    graph_title = "Feature space distribution"

    j = 0
    for i in uniq:
        # print(i)
        indx = group == i
        a = x1[indx]
        b = x2[indx]
        c = x3[indx]
        ax.plot(a, b, c, color=color[j], label=uniq[j], marker=marker[j], linestyle='', ms=7)
        j = j+1

    plt.xlabel('Weights_1', labelpad=10)
    plt.ylabel('Weights_2', labelpad=10)
    ax.set_zlabel('Weights_3', labelpad=10)
    plt.title(str(graph_title), fontsize=15)

    plt.legend(markerscale=20)
    plt.locator_params(nbins=6)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #plt.zticks(fontsize = 25)
    plt.legend(loc='upper left', frameon=False)
    plt.savefig(graph_name, bbox_inches='tight', dpi=400)
    plt.show()
    return ax, fig


def latent_animation(class_names, folder_created, train_results, train_labels, test_results, test_labels):
    graph_name = os.path.join(folder_created, 'Training_Feature_3D.png')
    ax, fig = Three_embeddings(train_results, train_labels, graph_name, class_names, ang=35)
    gif1_name = os.path.join(folder_created, 'Training_Feature_3D.gif')

    def rotate(angle):
        ax.view_init(azim=angle)

    angle = 3
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    ani.save(gif1_name, writer=animation.PillowWriter(fps=20))

    graph_name = os.path.join(folder_created, 'Testing_Feature_3D.png')
    ax, fig = Three_embeddings(test_results, test_labels, graph_name, class_names, ang=35)
    gif1_name = os.path.join(folder_created, 'Testing_Feature_3D.gif')

    def rotate(angle):
        ax.view_init(azim=angle)

    angle = 3
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, angle), interval=50)
    ani.save(gif1_name, writer=animation.PillowWriter(fps=20))
