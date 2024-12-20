# -*- coding: utf-8 -*-
"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Exploring Acoustic Emission Monitoring during Laser Powder Bed Fusion of premixed Ti6Al4V-Fe powder: Evidence of martensitic phase transformation supported by operando X-ray diffraction "
@any reuse of this code should be authorized by the first owner, code author

"""
# libraries to import
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
# import pywt
import pywt
import os


def filter(signal_window, sample_rate):
    """
    Applies a low-pass filter to the input signal window.

    Args:
        signal_window (array-like): The input signal window to be filtered.
        sample_rate (float): The sample rate of the input signal.

    Returns:
        array-like: The filtered signal window.

    """
    lowpass = 0.49*sample_rate  # Cut-off frequency of the filter
    lowpass_freq = lowpass / (sample_rate / 2)  # Normalize the frequency
    b, a = signal.butter(5, lowpass_freq, 'low')
    signal_window = signal.filtfilt(b, a, signal_window)

    #plt.plot(t, lowpassfilter, label='Low-Pass')
    return signal_window


def Wavelet2D(Material, sample_rate, rawspace, classspace, time, total_path):
    """
    Perform 2D wavelet analysis on the given data.

    Args:
        Material (str): The material name.
        sample_rate (int): The sample rate of the data.
        rawspace (pd.DataFrame): The raw data.
        classspace (pd.DataFrame): The class labels.
        time (np.array): The time values.
        total_path (str): The path to save the generated plots.

    Returns:
        None
    """

    classspace.columns = ['Categorical']
    data = pd.concat([rawspace, classspace], axis=1)

    # data = data.sample(frac=1, axis=0).reset_index(drop=True)

    print("Respective windows per category", data.Categorical.value_counts())
    # minval = min(data.Categorical.value_counts())
    minval = 1
    print("windows of the class: ", minval)
    data = pd.concat([data[data.Categorical == cat].head(minval)
                     for cat in data.Categorical.unique()])
    print("Balanced dataset: ", data.Categorical.value_counts())
    rawspace = data.iloc[:, :-1]
    rawspace = rawspace.to_numpy()
    classspace = data.iloc[:, -1]
    classspace = classspace.to_numpy()

    waveletname = 'morl'
    dt = time[1] - time[0]
    scale = 128
    scales = np.arange(1, scale)

    for i in range(len(classspace)):

        print(i)
        data = rawspace[i]
        data = filter(data, sample_rate)
        category = int(classspace[i])
        print(category)

        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams['agg.path.chunksize'] = len(data)

        # NFFT = 5000
        # dt = time[1] - time[0]
        [coefficients, frequencies] = pywt.cwt(data, scales, waveletname, dt)

        power = (abs(coefficients))
        lenthA = len(frequencies)
        # frequencies= frequencies[frequencies < lowpass]
        lenthB = len(frequencies)
        trimlenth = lenthA - lenthB
        power = np.delete(power, np.s_[0:trimlenth], axis=0)
        # power=np.log2(power)

        print(np.min(power))
        print(np.max(power))

        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_visible(True)
        im = plt.contourf(time, frequencies, power, cmap=plt.cm.rainbow)

        ax.axis('on')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.yaxis.offsetText.set_fontsize(20)
        ax.xaxis.offsetText.set_fontsize(20)
        ax.set_ylim(0, sample_rate/2)

        cb = plt.colorbar(im)
        cb.set_label(label='Intensity', fontsize=20)
        cb.ax.tick_params(labelsize=20)

        plottitle = str(Material)+'_'+str(category)
        plt.suptitle(plottitle, fontsize=20)

        plt.xlabel('Time(sec)', fontsize=20)
        plt.ylabel('Frequency(Hz)', fontsize=20)

        graphname = str(Material)+'_'+str(category)+'_2D_Wavelet.png'
        plt.savefig(os.path.join(total_path, graphname), bbox_inches='tight', dpi=800)

        plt.show()
        plt.clf()

        plt.figure(figsize=(6, 4))
        cwtmatr = signal.cwt(data, signal.ricker, scales)

        cwtmatr, _ = pywt.cwt(data, scales, waveletname, dt)
        plt.ylabel('Scales')
        plt.xlabel('Time(s)')
        plottitle = str(Material)+'_'+str(category)
        plt.suptitle(plottitle, fontsize=10)
        plt.imshow(cwtmatr, extent=[0, np.max(time), scale, 1], cmap='PRGn', aspect='auto',
                   vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
        ax = plt.gca()
        ax.invert_yaxis()
        plt.colorbar()
        graphname = str(Material)+'_'+str(category)+'_2D_CWT_Wavelet.png'
        plt.savefig(graphname, bbox_inches='tight', dpi=800)
        plt.show()
