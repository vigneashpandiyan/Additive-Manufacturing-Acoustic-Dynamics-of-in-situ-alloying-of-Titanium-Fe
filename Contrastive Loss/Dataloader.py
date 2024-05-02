# -*- coding: utf-8 -*-
"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in Laser Powder Bed Fusion Process"
@any reuse of this code should be authorized by the first owner, code author

"""
# libraries to import
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os
import pandas as pd
from torchvision import transforms
from sklearn.model_selection import train_test_split  # implementing train-test-split
import matplotlib.pyplot as plt


class contraster_dataset(Dataset):
    def __init__(self, df, train, transform=None):
        self.is_train = train
        self.transform = transform
        # self.to_pil = transforms.ToPILImage()

        if self.is_train:
            self.images = df.iloc[:, 1:].values.astype(np.uint8)
            self.labels = df.iloc[:, 0].values
            self.index = df.index.values
        else:
            self.images = df.iloc[:, 1:].values.astype(np.uint8)
            self.labels = df.iloc[:, 0].values
            self.index = df.index.values

    def __len__(self):

        return len(self.images)

    def __getitem__(self, item):
        #anchor_img = self.images[item].reshape(28, 28, 1)
        anchor_img = self.images[item]

        # print(item)
        should_get_same_class = random.randint(0, 1)
        if self.is_train:
            if should_get_same_class:
                label = self.labels[item]
                anchor_label = self.labels[item]
                # print(anchor_label)

                positive_list = self.index[self.index !=
                                           item][self.labels[self.index != item] == anchor_label]

                positive_item = random.choice(positive_list)
                positive_img = self.images[positive_item]

                anchor_label = 1

                return anchor_img, positive_img, anchor_label, label

            else:
                anchor_label = self.labels[item]
                label = self.labels[item]
                # print(anchor_label)
                negative_list = self.index[self.index !=
                                           item][self.labels[self.index != item] != anchor_label]
                negative_item = random.choice(negative_list)
                #negative_img = self.images[negative_item].reshape(28, 28, 1)
                negative_img = self.images[negative_item]
                anchor_label = 0

                return anchor_img, negative_img, anchor_label, label
        else:
            # if self.transform:
            #     anchor_img = self.transform(self.to_pil(anchor_img))
            label = self.labels[item]
            return anchor_img, label


def dataprocessing(df):
    database = df
    print(database.shape)
    database = database.apply(lambda x: (x - np.mean(x))/np.std(x), axis=1)
    # anomaly_database=anomaly_database.to_numpy().astype(np.float64)
    return database


def data_extract(datapath, Exptype):
    classfile = Exptype+'_Class_label.npy'
    classfile = os.path.join(datapath, classfile)
    rawfile = Exptype+'_Rawspace.npy'
    rawfile = os.path.join(datapath, rawfile)

    classspace = np.load(classfile).astype(np.int64)
    rawspace = np.load(rawfile).astype(np.float64)
    rawspace = pd.DataFrame(rawspace)
    rawspace = dataprocessing(rawspace)
    rawspace = rawspace.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        rawspace, classspace, test_size=0.20, random_state=66)

    Training = np.concatenate((y_train, X_train), axis=1)
    Testing = np.concatenate((y_test, X_test), axis=1)

    train_df = pd.DataFrame(Training)
    test_df = pd.DataFrame(Testing)

    train_df.head()
    return train_df, test_df


def data_plot(datapath, Exptype, folder_created):
    classfile = Exptype+'_Class_label.npy'
    classfile = os.path.join(datapath, classfile)
    rawfile = Exptype+'_Rawspace.npy'
    rawfile = os.path.join(datapath, rawfile)

    classspace = np.load(classfile).astype(np.int64)
    rawspace = np.load(rawfile).astype(np.float64)
    rawspace = pd.DataFrame(rawspace)
    rawspace = dataprocessing(rawspace)
    rawspace = rawspace.to_numpy()

    rawspace = pd.DataFrame(rawspace)
    classspace = pd.DataFrame(classspace)

    data = pd.concat([rawspace, classspace], axis=1)
    new_columns = list(data.columns)
    new_columns[-1] = 'target'
    data.columns = new_columns
    data.target.value_counts()
    data = data.sample(frac=1.0)
    class_names = ['Ti64', 'Ti64_3Fe', 'Ti64_6Fe']

    colour = ['green', 'red', 'blue', 'cyan', 'orange', 'purple']
    graphname = Exptype+'_data'+'_Visualize'+'.png'

    classes = data.target.unique()
    classes = np.sort(classes)
    fig, axs = plt.subplots(
        nrows=3,
        ncols=1,
        sharey=False,
        figsize=(8, 7),
        dpi=800
    )

    for i, cls in enumerate(classes):
        ax = axs.flat[i]
        df = data[data.target == cls].drop(labels='target', axis=1).mean(axis=0).to_numpy()
        plot_time_series(df, class_names[i], ax, colour[i], i)
    fig.tight_layout()
    plt.savefig(os.path.join(folder_created, graphname),
                bbox_inches='tight', pad_inches=0.1, dpi=800)
    plt.show()
    plt.clf()


def plot_time_series(data, class_name, ax, colour, i, n_steps=10):
    time_series_df = pd.DataFrame(data)

    smooth_path = time_series_df.rolling(n_steps).mean()
    path_deviation = 3 * time_series_df.rolling(n_steps).std()

    under_line = (smooth_path - path_deviation)[0]
    over_line = (smooth_path + path_deviation)[0]

    ax.plot(smooth_path, color=colour, linewidth=3)
    ax.fill_between(
        path_deviation.index,
        under_line,
        over_line,
        alpha=.450
    )
    ax.set_title(class_name)
    ax.set_ylim([-0.2, 0.2])
    ax.set_ylabel('Amplitude (V)')
    ax.set_xlabel('Window size (μs)')


def torch_loader(batch_size, train_df, test_df):
    train_ds = contraster_dataset(train_df,
                                  train=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor()
                                  ]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    test_ds = contraster_dataset(test_df,
                                 train=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor()
                                 ]))

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader
