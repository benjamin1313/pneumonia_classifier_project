# -*- coding: utf-8 -*-

from time import time
import numpy as np

import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, classification_report, f1_score
import os

#%% Load Data and format data

np.random.seed(42)

labels = ['NORMAL', 'PNEUMONIA']
img_size = 150
def get_training_data(data_dir):
    X = [] 
    y = []
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                X.append(resized_arr.flatten())
                y.append(class_num)
            except Exception as e:
                print(e)
    
    return np.array(X), np.array(y)

#%% PCA
def pca(n_components, X):
    # with max wished for n_componets 
    pca = PCA(n_components=n_components,svd_solver='randomized',whiten=True)
    pca.fit(X)

    pca.components_.reshape((n_components, img_size, img_size))

    X_new = pca.fit_transform(X)
    recovered = pca.inverse_transform(X_new)

    image = recovered[0].reshape(img_size, img_size)
    
    plt.figure()
    plt.imshow(image, cmap = matplotlib.cm.binary, interpolation="nearest")
    
    # max n_components
    pca = PCA(svd_solver='randomized',whiten=True)
    pca.fit(X)

    pca.components_.reshape((245, img_size, img_size)) # remember to change 245

    X_new = pca.fit_transform(X)
    recovered = pca.inverse_transform(X_new)

    image = recovered[0].reshape(img_size, img_size)
    
    plt.figure()
    plt.imshow(image, cmap = matplotlib.cm.binary, interpolation="nearest")

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_)/np.sum(pca.explained_variance_))
    plt.title('Explained variance - procentdel')
    plt.xlabel('Dimension nummer')

#%% ML models
def train_decision_tree(X, y):
    clf = DecisionTreeClassifier()
    clf.fit(X,y)    
    return clf
    
#%% main
if __name__ == "__main__":
    X_train, y_train = get_training_data('chest_xray/chest_xray/train')
    X_test, y_test = get_training_data('chest_xray/chest_xray/test')
    X_val, y_val = get_training_data('chest_xray/chest_xray/val')


#%%
    shuffle_index = np.random.permutation(245)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#%% Plots example image
    cv2.imshow("image", X_train[0].reshape(img_size,img_size))

#%%
    pca(64, X_train)
    
#%%
    tree = train_decision_tree(X_train, y_train)
    y_pred = tree.predict(X_test)
    MSE = mean_squared_error(y_pred, y_test)
    print(f'RMSE = {MSE**(1/2)}')
    print(f'F1: {f1_score(y_test, y_pred)}')
    