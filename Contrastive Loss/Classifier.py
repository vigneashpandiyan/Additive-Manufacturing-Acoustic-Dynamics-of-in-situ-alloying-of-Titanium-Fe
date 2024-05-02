# -*- coding: utf-8 -*-
"""

@author: srpv
contact: vigneashwara.solairajapandiyan@empa.ch, vigneashpandiyan@gmail.com

The codes in this following script will be used for the publication of the following work

"Qualify-As-You-Go: Sensor Fusion of Optical and Acoustic Signatures with Contrastive Deep Learning for Multi-Material Composition Monitoring in Laser Powder Bed Fusion Process"
@any reuse of this code should be authorized by the first owner, code author

"""
# libraries to import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import os
from sklearn import metrics
# import pydot
import collections
# import pydotplus
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
import joblib
from sklearn.model_selection import cross_val_score
from IPython.display import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # implementing train-test-split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
# from Plots import *

# %%


def classifier_linear(Exptype, folder_created):
    train_embeddings = Exptype+'_train_embeddings'+'_' + '.npy'
    train_embeddings = os.path.join(folder_created, train_embeddings)

    train_labelsname = Exptype+'_train_labels'+'_'+'.npy'
    train_labelsname = os.path.join(folder_created, train_labelsname)

    test_embeddings = Exptype+'_test_embeddings'+'_' + '.npy'
    test_embeddings = os.path.join(folder_created, test_embeddings)

    test_labelsname = Exptype+'_test_labels'+'_'+'.npy'
    test_labelsname = os.path.join(folder_created, test_labelsname)

    X_train = np.load(train_embeddings).astype(np.float64)
    y_train = np.load(train_labelsname).astype(np.float64)
    X_test = np.load(test_embeddings).astype(np.float64)
    y_test = np.load(test_labelsname).astype(np.float64)

    y_pred_prob, pred_prob = LR(X_train, X_test, y_train, y_test, folder_created)


def LR(X_train, X_test, y_train, y_test, folder_created):

    model = LogisticRegression(max_iter=1000, random_state=123)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    pred_prob = model.predict_proba(X_test)
    y_pred_prob = np.vstack((y_test, predictions)).transpose()

    y_pred_prob = np.hstack((y_pred_prob, pred_prob))

    print("LogisticRegression Accuracy:", metrics.accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    graph_name1 = 'LR'+'_without normalization w/o Opt'
    graph_name2 = 'Logistic Regression'

    graph_1 = 'LR_Confusion_Matrix_No_Opt.png'
    graph_2 = 'LR_Confusion_Matrix_Opt.png'

    titles_options = [(graph_name1, None, graph_1),
                      (graph_name2, 'true', graph_2)]

    for title, normalize, graphname in titles_options:
        plt.figure(figsize=(20, 10), dpi=400)
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                                     display_labels=[
                                                         'Ti64', 'Ti64_3Fe', 'Ti64_6Fe'],
                                                     cmap=plt.cm.Reds, xticks_rotation='vertical',
                                                     normalize=normalize)
        plt.title(title, size=12)

        plt.savefig(os.path.join(folder_created, graphname), bbox_inches='tight', dpi=400)
    savemodel = os.path.join(folder_created, 'LR_model.sav')
    joblib.dump(model, savemodel)

    return y_pred_prob, pred_prob
