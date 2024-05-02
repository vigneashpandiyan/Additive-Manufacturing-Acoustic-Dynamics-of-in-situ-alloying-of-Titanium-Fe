# -*- coding: utf-8 -*-
"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Exploring Acoustic Emission Monitoring during Laser Powder Bed Fusion of premixed Ti6Al4V-Fe powder: Evidence of martensitic phase transformation supported by operando X-ray diffraction "
@any reuse of this code should be authorized by the first owner, code author

"""
# libraries to import
import random
from matplotlib import cm
import pywt
from operator import itemgetter
from itertools import groupby
from scipy import signal
import os
from pathlib import Path
import ntpath
import re
import matplotlib.pyplot as plt
import pandas as pd
import glob
import numpy as np

import glob
import seaborn as sns
import matplotlib.pyplot as mpl
from Utils import *
mpl.rcParams['agg.path.chunksize'] = 1000000


def path_leaf(path):
    """
    Extracts the file name from a given path.

    Args:
        path (str): The path from which to extract the file name.

    Returns:
        str: The file name extracted from the path.
    """
    head, tail = ntpath.split(path)
    return tail
    # return tail or ntpath.basename(head)

# %%


# path to the data
path = r'C:\Users\srpv\Desktop\Git\Additive-Manufacturing-Acoustic-Dynamics-of-in-situ-alloying-of-Titanium-Fe\ML classifier\Crack dynamics\Data\KM_20_Ti64_6Fe'
sample_rate = 1500000  # sample rate of the data

isDirectory = os.path.isdir(path)

file = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
total_path = os.path.dirname(file)
print(total_path)

if isDirectory:
    print(path)
    print(isDirectory)
    Channel0 = (os.path.join(path, 'channel_0_decoded'))
    print(Channel0)
    Channel1 = (os.path.join(path, 'channel_1_decoded'))
    print(Channel1)

    all_files = glob.glob(Channel0 + "/*.csv")

    for filename in all_files:
        tail = path_leaf(filename)
        A = tail.split('.')
        print(A)
        data_file_name = A[0]
        print(data_file_name)

        # Create a folder to save the data
        folder_created = os.path.join(total_path, data_file_name)
        print(folder_created)
        try:
            os.makedirs(folder_created, exist_ok=True)
            print("Directory created....")
        except OSError as error:
            print("Directory already exists....")

        Channel_0 = (os.path.join(Channel0, tail))
        print(Channel0)
        Channel_1 = (os.path.join(Channel1, tail))
        print(Channel1)

        data1 = pd.read_csv(Channel_0)
        data1 = data1.to_numpy()
        data1 = np.ravel(data1)
        data2 = pd.read_csv(Channel_1)
        data2 = data2.to_numpy()
        data2 = np.ravel(data2)

        length = len(data1)
        N = length
        t0 = 0
        dt = 1/sample_rate
        time = np.arange(0, N) * dt + t0

        # Plot the raw data
        uniq = str(random.randint(0, 9999))
        plt.rcParams.update(plt.rcParamsDefault)
        fig = plt.figure(figsize=(10, 6), dpi=800)
        fig, ((ax1, ax2)) = plt.subplots(2, 1)
        title = ' AE '+'Signal'
        fig.suptitle(title)
        plt.rc("font", size=8)

        ax1.plot(time[0:800000], data1[0:800000], 'red', linewidth=2.5, label='Laser')
        ax1.legend(fontsize=15)
        ax1.set_ylabel('Voltage (V)')
        ax2.plot(time[0:800000], data2[0:800000], 'blue', linewidth=1, label='AE')
        ax2.legend(fontsize=15)
        # ax1.set_xlim(0,0.3)
        ax2.set_xlabel('Time (sec)', labelpad=5)
        ax2.set_ylabel('Voltage (V)')
        # ax2.set_xlim(0,0.3)

        # Plot the raw data
        for ax in fig.get_axes():
            ax.label_outer()
        # plt.xlim((0, 40e5))

        tail = path_leaf(path)
        A = tail.split('.')
        data_file_name = A[0]
        graph_2 = str(data_file_name)+'_'+uniq+'_'+'RawAE'+'signal'+'.png'
        plt.savefig(os.path.join(folder_created, graph_2), bbox_inches='tight', dpi=200)
        plt.show()

        data1_ = data1[240000:320000]  # truncating from whole data
        data2_ = data2[240000:320000]  # truncating from whole data
        length = len(data1_)
        N = length
        t0 = 0
        dt = 1/sample_rate
        time_ = np.arange(0, N) * dt + t0

        plt.rcParams.update(plt.rcParamsDefault)
        fig = plt.figure(figsize=(10, 6), dpi=800)
        fig, ((ax1, ax2)) = plt.subplots(2, 1)
        title = ' AE '+'Signal'
        fig.suptitle(title)
        plt.rc("font", size=8)

        ax1.plot(time_, data1_, 'red', linewidth=2.5, label='Laser')
        ax1.legend(fontsize=15)
        ax1.set_ylabel('Voltage (V)')
        ax2.plot(time_, data2_, 'blue', linewidth=1, label='AE')
        ax2.legend(fontsize=15)
        # ax1.set_xlim(0,0.3)
        ax2.set_xlabel('Time (sec)', labelpad=5)
        ax2.set_ylabel('Voltage (V)')
        # ax2.set_xlim(0,0.3)

        for ax in fig.get_axes():
            ax.label_outer()
        # plt.xlim((0, 40e5))

        tail = path_leaf(path)
        A = tail.split('.')
        data_file_name = A[0]
        graph_2 = str(data_file_name)+'_'+uniq+'_'+'AE'+'signal'+'.png'
        plt.savefig(os.path.join(folder_created, graph_2), bbox_inches='tight', dpi=200)
        plt.show()

        # Plot the spectrogram
        specgram3d(data_file_name, uniq, data2_, sample_rate, time_, folder_created)
