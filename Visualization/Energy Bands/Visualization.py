import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from IPython.display import Image

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.pyplot import specgram
import seaborn as sns
from scipy.stats import norm


from matplotlib import cm
from scipy import signal
import pywt
import matplotlib.patches as mpatches
from matplotlib import colors
# %%

sns.set(font_scale=1.3)
sns.set_style("whitegrid", {'axes.grid': False})
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
sample_rate = 1500000
windowsize = 5000


total_path = r'C:\Users\srpv\Desktop\Git\Additive-Manufacturing-Acoustic-Dynamics-of-in-situ-alloying-of-Titanium-Fe\ML classifier\Feature extraction'
print(total_path)

# %%


def Histplotsplit(data, Material, name):

    df = data
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=800, sharex=False, sharey=True)
    sns.set(font_scale=1.5)
    colors = ['#2171b5', '#980043', '#006d2c']
    plt.rcParams.update(plt.rcParamsDefault)

    for i in range(3):

        print(i)

        if i == 0:
            j = 0
        elif i == 1:
            j = 1
        elif i == 2:
            j = 2

        print("new value: ", j)
        x = df.loc[df.categorical == df.categorical.unique()[i], 'Feature']
        p = sns.kdeplot(x, shade=True, alpha=.5, color=colors[i], label=str(
            df.categorical.unique()[i]), ax=axes[j], edgecolors="k")
        p.set_xlabel(str(name)+" (AE energy)", fontsize=20)
        p.set_ylabel("Relative density", fontsize=20)
        ax = axes[j]
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.tick_params(axis='both', which='minor', labelsize=25)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        # ax.legend(fontsize=30,bbox_to_anchor=(0.95, 1.50))

    plt.ylabel("Relative frequency")
    plt.rc("font", size=23)
    plt.tight_layout()

    plot = str(Material)+'_'+str(name)+'_Histplotsplit.jpg'
    plt.savefig(plot, bbox_inches='tight', dpi=800)
    plt.show()


# %%

def plots(i, Featurespace, classspace, Material, name):
    Featurespace = Featurespace.transpose()
    data = (Featurespace[i])
    data = data.astype(np.float64)
    #data= abs(data)
    df1 = pd.DataFrame(data)
    df1.rename(columns={df1.columns[0]: "Feature"}, inplace=True)
    df2 = pd.DataFrame(classspace)

    df2 = pd.DataFrame(classspace)
    df2.columns = ['Categorical']

    df2 = df2['Categorical'].replace(0, 'Ti64')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(1, 'Ti64_3Fe')
    df2 = pd.DataFrame(df2)
    df2 = df2['Categorical'].replace(2, 'Ti64_6Fe')
    df2 = pd.DataFrame(df2)

    df2.rename(columns={df2.columns[0]: "categorical"}, inplace=True)
    data = pd.concat([df1, df2], axis=1)

    Histplotsplit(data, Material, name)

    return data


# %%
    # 4#7
M = ['KM', 'CM']

for Material in M:

    classfile = str(Material)+'_Class_label'+'.npy'
    featurefile = str(Material)+'_PSD'+'.npy'
    featurefile_1 = (os.path.join(total_path, featurefile))
    classfile_1 = (os.path.join(total_path, classfile))
    Featurespace = np.load(featurefile_1).astype(np.float64)
    classspace = np.load(classfile_1).astype(np.float64)

    Featurespace = pd.DataFrame(Featurespace)
    classspace = pd.DataFrame(classspace)

    classspace.columns = ['Categorical']

    data = pd.concat([Featurespace, classspace], axis=1)
    minval = min(data.Categorical.value_counts())

    data = pd.concat([data[data.Categorical == cat].head(minval)
                     for cat in data.Categorical.unique()])
    Featurespace = data.iloc[:, :-1]
    classspace = data.iloc[:, -1]

    values, counts = np.unique(classspace, return_counts=True)
    print(values, counts)

    classspace = classspace.to_numpy()
    Featurespace = Featurespace.to_numpy()

    data = plots(0, Featurespace, classspace, Material, "0-150 kHz")
    data = plots(1, Featurespace, classspace, Material, "150-300 kHz")
    data = plots(2, Featurespace, classspace, Material, "300-450 kHz")
    data = plots(3, Featurespace, classspace, Material, "450-600 kHz")
    data = plots(4, Featurespace, classspace, Material, "600-750 kHz")
