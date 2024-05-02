# -*- coding: utf-8 -*-
"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Exploring Acoustic Emission Monitoring during Laser Powder Bed Fusion of premixed Ti6Al4V-Fe powder: Evidence of martensitic phase transformation supported by operando X-ray diffraction "
@any reuse of this code should be authorized by the first owner, code author

"""
# libraries to import
from matplotlib import mlab
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
# import pywt
import pywt
import os
from matplotlib import cm


def filter(signal_window, sample_rate):
    """
    Applies a low-pass filter to the given signal window.

    Args:
        signal_window (array-like): The input signal window to be filtered.
        sample_rate (float): The sample rate of the signal.

    Returns:
        array-like: The filtered signal window.

    """
    lowpass = 0.49 * sample_rate  # Cut-off frequency of the filter
    lowpass_freq = lowpass / (sample_rate / 2)  # Normalize the frequency
    b, a = signal.butter(5, lowpass_freq, 'low')
    signal_window = signal.filtfilt(b, a, signal_window)

    #plt.plot(t, lowpassfilter, label='Low-Pass')
    return signal_window


def specgram3d(Material, uniq, data, sample_rate, time, folder_created, ax=None, title=None):
    """
    Generate a 3D spectrogram plot.

    Parameters:
    - Material (str): The material name.
    - uniq (str): The unique identifier.
    - data (ndarray): The input data.
    - sample_rate (int): The sample rate of the data.
    - time (float): The time duration of the data.
    - folder_created (str): The path of the folder where the plot will be saved.
    - ax (Axes3D, optional): The 3D axes object. Defaults to None.
    - title (str, optional): The title of the plot. Defaults to None.

    Returns:
    - X (ndarray): The time values.
    - Y (ndarray): The frequency values.
    - Z (ndarray): The amplitude values in dB.
    """

    fig = plt.figure(figsize=(7, 12))
    ax = plt.axes(projection='3d')
    ax.view_init(azim=40, elev=70)

    spec, freqs, t = mlab.specgram(data, Fs=sample_rate)
    X, Y, Z = t[None, :], freqs[:, None],  20*np.log10(spec)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_xlabel('Time (sec)', labelpad=7)
    ax.set_ylabel('Frequency (Hz)', labelpad=9)
    ax.set_zlabel('Amplitude (dB)', labelpad=9)

    # ax.set_title('Surface plot')
    ax.grid(False)
    ax.set_facecolor('white')

    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

    ax.yaxis.set_major_formatter('{x:0.2f}')

    fig.colorbar(surf, ax=ax,
                 shrink=0.1, aspect=4)

    graphname = str(Material)+'_'+str(uniq)+'_2D_Wavelet.png'

    plt.savefig(os.path.join(folder_created, graphname), bbox_inches='tight', dpi=800)

    plt.show()
    return X, Y, Z
