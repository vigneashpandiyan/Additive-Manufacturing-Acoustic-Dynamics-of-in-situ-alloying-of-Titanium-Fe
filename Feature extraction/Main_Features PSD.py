import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import scipy.signal as signal

from scipy.stats import kurtosis, skew
from scipy.signal import welch, periodogram
from numpy.fft import fftshift, fft
from scipy.signal import find_peaks
import statistics
from scipy import stats
from collections import Counter
from scipy.stats import entropy
from scipy.signal import hilbert, chirp
from scipy.stats import entropy
import os
#import librosa
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
def filter(signal_window):
    lowpass = 0.49*sample_rate  # Cut-off frequency of the filter
    lowpass_freq = lowpass / (sample_rate / 2)  # Normalize the frequency
    b, a = signal.butter(5, lowpass_freq, 'low')
    lowpassfilter = signal.filtfilt(b, a, signal_window)
    #plt.plot(t, lowpassfilter, label='Low-Pass')
    return lowpassfilter


# %%

def get_band(band_size, band_max_size):
    band_window = 0
    band = []
    for y in range(band_size):
        band.append(band_window)
        band_window += band_max_size / band_size
    return band


def spectrumpower(psd, band, freqs):
    length = len(band)
    # print(length)
    Feature_deltapower = []
    Feature_relativepower = []
    for i in range(band_size-1):
        if i <= (len(band)):
            ii = i
            # print('low frequencies :',band[ii])
            low = band[ii]

            ii = i+1
            high = band[ii]
            # print('high frequencies :',band[ii])

            #freq_res = freqs[high] - freqs[low]
            idx_delta = np.logical_and(freqs >= low, freqs <= high)
            total_power = sum(psd)
            delta_power = sum(psd[idx_delta])
            delta_rel_power = delta_power / total_power
            Feature_deltapower.append(delta_power)

    return Feature_deltapower


# %%

def function(val):

    i = 0

    signal_window = filter(val)
    win = 4 * sample_rate
    freqs, psd = periodogram(signal_window, sample_rate, window='hamming')
    band_max_size = 900000
    band = get_band(band_size, band_max_size)

    # print(band)
    # PSD absolute and relative power in each band 10 Features
    Feature1 = spectrumpower(psd, band, freqs)
    Feature = np.asarray(Feature1)

    if i == 0:
        #     print("--reached")
        size_of_Feature_vectors = int(len(Feature))
        size_of_dataset = int(len(signal_window))

        Feature_vectors = np.empty((0, size_of_Feature_vectors))
        rawdataset = np.empty((0, size_of_dataset))

    # print(label)
    Feature_vectors = np.append(Feature_vectors, [Feature], axis=0)
    rawdataset = np.append(rawdataset, [signal_window], axis=0)

    return Feature_vectors

# %%


def myfunction(data_new):
    columnsdata = data_new.transpose()
    columns = np.atleast_2d(columnsdata).shape[1]
    featurelist = []
    classlist = []
    rawlist = []

    # for row in loop:
    for k in range(columns):

        val = columnsdata[:, k]
        # totaldatapoints= val.size
        # window=round(totaldatapoints/windowsize)
        Feature_vectors = function(val)

        print(k)
        # print(Feature_vectors)

        for item in Feature_vectors:

            featurelist.append(item)

    return featurelist


# %%

M = ['KM', 'CM']

for Materials in M:

    class_filename = str(Materials)+'_'+'Class_label'+'.npy'
    path_ = os.path.join(path, class_filename)
    Class_label = np.load(path_).astype(np.float64)
    np.save(class_filename, Class_label, allow_pickle=True)

    path_ = str(Materials)+'_'+'Rawspace'+'.npy'
    path_ = os.path.join(path, path_)
    data = np.load(path_).astype(np.float64)
    featurelist = myfunction(data)
    Featurespace = np.asarray(featurelist)
    Featurespace = Featurespace.astype(np.float64)

    data_filename = str(Materials)+'_PSD'+'.npy'
    np.save(data_filename, Featurespace, allow_pickle=True)
