# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 20:55:13 2023

@author: srpv
"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
import os
from Utils import *
# %%

file = os.path.join(os.getcwd(), os.listdir(os.getcwd())[0])
total_path = os.path.dirname(file)
print(total_path)

# %%
sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid': False})
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
sample_rate = 1500000
windowsize = 5000
t0 = 0
dt = 1/sample_rate
time = np.arange(0, windowsize) * dt + t0

path = r'C:\Users\srpv\Desktop\Git\Additive-Manufacturing-Acoustic-Dynamics-of-in-situ-alloying-of-Titanium-Fe\ML classifier\Data_preprocessing'


M = ['KM', 'CM']

for Material in M:

    classfile = str(Material)+'_Class_label'+'.npy'
    rawfile = str(Material)+'_Rawspace'+'.npy'

    classfile = os.path.join(path, classfile)
    rawfile = os.path.join(path, rawfile)

    classspace = np.load(classfile).astype(np.float64)
    classspace = pd.DataFrame(classspace)

    rawspace = np.load(rawfile).astype(np.float64)
    rawspace = pd.DataFrame(rawspace)
    # Frequencyplot(rawspace, classspace, Material)

    Wavelet2D(Material, sample_rate, rawspace, classspace, time, total_path)
