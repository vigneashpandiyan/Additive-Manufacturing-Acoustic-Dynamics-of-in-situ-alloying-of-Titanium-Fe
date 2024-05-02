from sklearn.model_selection import train_test_split  # implementing train-test-split
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import seaborn as sns
from scipy.stats import norm
# import joypy
import pandas as pd
from matplotlib import cm
from scipy import signal
import pywt
import matplotlib.patches as mpatches
import os

# %%

datapath = r'C:\Users\srpv\Desktop\Git\Additive-Manufacturing-Acoustic-Dynamics-of-in-situ-alloying-of-Titanium-Fe\ML classifier\Rawdata'

# %%

sns.set(font_scale=1.5)
sns.set_style("whitegrid", {'axes.grid': False})
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
sample_rate = 1500000
windowsize = 5000


def dataprocessing(df):
    database = df
    print(database.shape)
    database = database.apply(lambda x: (x - np.mean(x))/np.std(x), axis=1)
    # anomaly_database=anomaly_database.to_numpy().astype(np.float64)
    return database


def data(case, exp):

    rawfile = str(case)+'-'+str(exp)+'.npy'
    classfile = 'Classlabel_'+str(case)+'-'+str(exp)+'.npy'

    rawspace = os.path.join(datapath, rawfile)
    print(rawspace)
    rawspace = np.load(rawspace).astype(np.float64)

    classspace = os.path.join(datapath, classfile)
    print(classspace)
    classspace = np.load(classspace)

    rawspace = pd.DataFrame(rawspace)
    rawspace = dataprocessing(rawspace)
    rawspace = rawspace.to_numpy()

    return rawspace, classspace


Datasets = ['CM', 'KM']

for Exptype in Datasets:

    Raw_1, label_1 = data(Exptype, 'Ti64')
    Raw_2, label_2 = data(Exptype, 'Ti64_3Fe')
    Raw_3, label_3 = data(Exptype, 'Ti64_6Fe')

    rawspace = np.concatenate((Raw_1, Raw_2, Raw_3), axis=0)
    classspace = np.concatenate((label_1, label_2, label_3), axis=0)
    classspace = np.expand_dims(classspace, axis=1)

    # %%
    # Visualization

    X_train, X_test, y_train, y_test = train_test_split(
        rawspace, classspace, test_size=0.25, random_state=66)

    Training = np.concatenate((y_train, X_train), axis=1)
    Testing = np.concatenate((y_test, X_test), axis=1)

    classfile = Exptype+'_Class'+'_' + 'label'+'.npy'
    classspace = classspace.astype(np.float64)
    np.save(classfile, classspace, allow_pickle=True)

    rawfile = Exptype+'_Rawspace'+'.npy'
    rawspace = rawspace.astype(np.float64)
    np.save(rawfile, rawspace, allow_pickle=True)
