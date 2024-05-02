# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:18:58 2023

@author: srpv
"""
import os
import numpy as np
from Utils_Featureextraction import *
print(np.__version__)

# %%

sample_rate = 1500000
windowsize = 5000
t0 = 0
dt = 1/sample_rate
time = np.arange(0, windowsize) * dt + t0
band_size = 6
peaks_to_count = 7
count = 0

path = r'C:\Users\srpv\Desktop\Git\Additive-Manufacturing-Acoustic-Dynamics-of-in-situ-alloying-of-Titanium-Fe\ML classifier\Data_preprocessing'

# %%


def Time_frequency_feature(Materials, sample_rate, band_size, peaks_to_count):

    path_ = str(Materials)+'_'+'Rawspace'+'.npy'
    path_ = os.path.join(path, path_)
    data = np.load(path_).astype(np.float64)
    Feature_vectors = feature_extraction(data, sample_rate, band_size, peaks_to_count)

    return Feature_vectors


M = ['KM', 'CM']

for Materials in M:
    featurelist = Timeseries_feature(Materials, sample_rate, band_size, peaks_to_count)
    Featurespace = np.asarray(featurelist)
    Featurespace = Featurespace.astype(np.float64)
    featurefile = str(Materials)+'_Featurespace'+'_' + str(windowsize)+'.npy'
    np.save(featurefile, Featurespace, allow_pickle=True)


# %%
