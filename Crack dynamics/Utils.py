# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:57:22 2024

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch


The codes in this following script will be used for the publication of the following work

"Classification of Progressive Wear on a Multi-Directional Pin-on-Disc 
Tribometer Simulating Conditions in Human Joints-UHMWPE against CoCrMo 
Using Acoustic Emission and Machine Learning"

@any reuse of this code should be authorized by the first owner, code author
"""

# %%
# Libraries to import

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

    lowpass = 0.49 * sample_rate  # Cut-off frequency of the filter
    lowpass_freq = lowpass / (sample_rate / 2)  # Normalize the frequency
    b, a = signal.butter(5, lowpass_freq, 'low')
    signal_window = signal.filtfilt(b, a, signal_window)

    #plt.plot(t, lowpassfilter, label='Low-Pass')
    return signal_window


# def Wavelet2D(Material, uniq, data, sample_rate, time):

#     data = filter(data, sample_rate)

#     waveletname = 'morl'
#     dt = time[1] - time[0]
#     scales = np.arange(1, 100)

#     plt.rcParams.update(plt.rcParamsDefault)
#     plt.rcParams['agg.path.chunksize'] = len(data)

#     # NFFT = 5000
#     # dt = time[1] - time[0]
#     [coefficients, frequencies] = pywt.cwt(data, scales, waveletname, dt)

#     power = (abs(coefficients))
#     lenthA = len(frequencies)
#     # frequencies= frequencies[frequencies < lowpass]
#     lenthB = len(frequencies)
#     trimlenth = lenthA - lenthB
#     power = np.delete(power, np.s_[0:trimlenth], axis=0)
#     # power=np.log2(power)

#     print(np.min(power))
#     print(np.max(power))

#     fig, ax = plt.subplots(figsize=(12, 7))
#     fig.patch.set_visible(True)
#     im = plt.contourf(time, frequencies, power, cmap=plt.cm.rainbow)

#     ax.axis('on')
#     ax.tick_params(axis='both', which='major', labelsize=20)
#     ax.tick_params(axis='both', which='minor', labelsize=20)
#     ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
#     ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
#     ax.yaxis.offsetText.set_fontsize(20)
#     ax.xaxis.offsetText.set_fontsize(20)
#     ax.set_ylim(0, sample_rate/2)

#     cb = plt.colorbar(im)
#     cb.set_label(label='Intensity', fontsize=20)
#     cb.ax.tick_params(labelsize=20)

#     plottitle = str(Material)
#     plt.suptitle(plottitle, fontsize=20)

#     plt.xlabel('Time(sec)', fontsize=20)
#     plt.ylabel('Frequency(Hz)', fontsize=20)

#     graphname = str(Material)+'_'+str(uniq)+'_2D_Wavelet.png'
#     plt.savefig(graphname, bbox_inches='tight', dpi=800)

#     plt.show()
#     plt.clf()


# def surface_waveletplot(data, time):
#     scales = np.arange(1, 100)
#     waveletname = 'morl'
#     cmap = plt.cm.jet
#     dt = time[1] - time[0]
#     [coefficients, frequencies] = pywt.cwt(data, scales, waveletname, dt)
#     power = (abs(coefficients)) ** 2
#     period = 1. / frequencies
#     lenthA = len(frequencies)
#     frequencies = frequencies[frequencies < 100000]
#     #frequencies= frequencies[frequencies > 10000]

#     lenthB = len(frequencies)
#     trimlenth = lenthA - lenthB
#     power = np.delete(power, np.s_[0:trimlenth], axis=0)
#     #freq=pywt.scale2frequency(waveletname, scales, precision=8)
#     timeplot, frequencies = np.meshgrid(time, frequencies)
#     return frequencies, power, timeplot


# def ThreeDwaveletplot(Material, uniq, data, sample_rate, time):

#     data = filter(data, sample_rate)

#     frequencies, power, timeplot = surface_waveletplot(data, time)
#     X = timeplot
#     Y = frequencies
#     Z = power

#     fig = plt.figure(figsize=(14, 10), dpi=200)
#     ax = fig.add_subplot(projection='3d')

#     # plot_trisurf,plot_surface
#     surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                            linewidth=0, rstride=1, cstride=1, antialiased=True, vmin=np.min(Z), vmax=np.max(Z))  # rstride=25, cstride=1,
#     vmax = np.max(Z)
#     ax.set_xlim(0, np.max(X))
#     # ax.set_ylim(0, sample_rate//2)
#     ax.set_zlim(0, vmax)
#     ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
#     ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
#     ax.ticklabel_format(axis='z', style='sci', scilimits=(0, 0))
#     plottitle = str(Material)+'_'+str(uniq)
#     plt.title(plottitle, loc='center')

#     ax.zaxis.set_rotate_label(False)
#     plt.xlabel('Time', labelpad=5)
#     plt.ylabel('Frequency', labelpad=5)
#     ax.set_zlabel('Power', labelpad=5, rotation=90)
#     ax.grid(False)
#     ax.set_facecolor('white')

#     ax.w_xaxis.pane.fill = False
#     ax.w_yaxis.pane.fill = False
#     ax.w_zaxis.pane.fill = False

#     ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

#     ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
#     ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

#     ax.view_init(azim=10, elev=30)
#     #cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
#     fig.colorbar(surf, shrink=0.4, aspect=10, ax=ax)
#     fig.patch.set_visible(True)
#     #plt.tight_layout(pad=4, w_pad=10, h_pad=1.0)
#     plt.locator_params(nbins=4)
#     graph_1 = str(Material)+'_'+str(uniq)+'_3DWavelet.png'
#     plt.savefig(graph_1, bbox_inches='tight', dpi=800)
#     plt.show()
#     plt.clf()


# def STFT(Material, uniq, data, sample_rate, time, folder_created):
#     f, t, Sxx = signal.spectrogram(data, sample_rate)

#     fig = plt.figure(figsize=(7, 12), dpi=200)
#     plt.pcolormesh(t, f, Sxx, shading='gouraud')

#     plottitle = str(Material)
#     plt.suptitle(plottitle, fontsize=20)

#     plt.xlabel('Time(sec)', fontsize=20)
#     plt.ylabel('Frequency(Hz)', fontsize=20)

#     graphname = str(Material)+'_'+str(uniq)+'_2D_Wavelet.png'
#     plt.savefig(os.path.join(folder_created, graphname), bbox_inches='tight', dpi=800)

#     plt.show()
#     plt.clf()


def specgram3d(Material, uniq, data, sample_rate, time, folder_created, ax=None, title=None):

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

    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    # ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(5, 5))
    # ax.ticklabel_format(axis='z', style='sci', scilimits=(0, 0))

    # plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    fig.colorbar(surf, ax=ax,
                 shrink=0.1, aspect=4)

    graphname = str(Material)+'_'+str(uniq)+'_2D_Wavelet.png'

    plt.savefig(os.path.join(folder_created, graphname), bbox_inches='tight', dpi=800)

    plt.show()
    return X, Y, Z


# def phase_spectrum(Material, uniq, data, sample_rate, time, ax=None, title=None):

#     fig, ax = plt.subplots(figsize=(12, 7))
#     ax.phase_spectrum(data, Fs=sample_rate, color="green")
#     plt.show()
# fig2, ax2 = plt.subplots(subplot_kw={'projection': '3d'})
# specgram3d(x, srate=fs, ax=ax2)
# plt.show()
