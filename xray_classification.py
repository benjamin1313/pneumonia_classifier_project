# -*- coding: utf-8 -*-

from time import time
import numpy as np

import cv2
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import os

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf





# taken from gridsearch exercise 
def SearchReport(model): 
    
    def GetBestModelCTOR(model, best_params):
        def GetParams(best_params):
            ret_str=""          
            for key in sorted(best_params):
                value = best_params[key]
                temp_str = "'" if str(type(value))=="<class 'str'>" else ""
                if len(ret_str)>0:
                    ret_str += ','
                ret_str += f'{key}={temp_str}{value}{temp_str}'  
            return ret_str          
        try:
            param_str = GetParams(best_params)
            return type(model).__name__ + '(' + param_str + ')' 
        except:
            return "N/A(1)"
        
    print("\nBest model set found on train set:")
    print()
    print(f"\tbest parameters={model.best_params_}")
    print(f"\tbest '{model.scoring}' score={model.best_score_}")
    print(f"\tbest index={model.best_index_}")
    print()
    print(f"Best estimator CTOR:")
    print(f"\t{model.best_estimator_}")
    print()
    try:
        print(f"Grid scores ('{model.scoring}') on development set:")
        means = model.cv_results_['mean_test_score']
        stds  = model.cv_results_['std_test_score']
        i=0
        for mean, std, params in zip(means, stds, model.cv_results_['params']):
            print("\t[%2d]: %0.3f (+/-%0.03f) for %r" % (i, mean, std * 2, params))
            i += 1
    except:
        print("WARNING: the random search do not provide means/stds")
    
    global currmode
    currmode = "xray"           
    assert "f1_micro"==str(model.scoring), f"come on, we need to fix the scoring to be able to compare model-fits! Your scoreing={str(model.scoring)}...remember to add scoring='f1_micro' to the search"   
    return f"best: dat={currmode}, score={model.best_score_:0.5f}, model={GetBestModelCTOR(model.estimator,model.best_params_)}", model.best_estimator_ 

def ClassificationReport(model, X_test, y_test, target_names=None):
    assert X_test.shape[0]==y_test.shape[0]
    print("\nDetailed classification report:")
    print("\tThe model is trained on the full development set.")
    print("\tThe scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, model.predict(X_test)                 
    print(classification_report(y_true, y_pred, target_names))
    print()
    
def FullReport(model, X_test, y_test, t):
    print(f"SEARCH TIME: {t:0.2f} sec")
    beststr, bestmodel = SearchReport(model)
    ClassificationReport(model, X_test, y_test)    
    print(f"CTOR for best model: {bestmodel}\n")
    print(f"{beststr}\n")
    return beststr, bestmodel








#%% Load Data and format data

np.random.seed(42)

labels = ['NORMAL', 'PNEUMONIA']
img_size = 150
def load_data(data_dir):
    X = [] 
    y = []
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                X.append(resized_arr) # use X.append(resized_arr.flatten()) for enything else that the CNN
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

    pca.components_.reshape((245, img_size, img_size)) # remember to change 245 if max n_components change

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
    
def train_random_forrest(X, y): #best parameters={'criterion': 'gini', 'n_estimators': 167}

    clf = RandomForestClassifier(n_jobs=-1)
    
    tuning_parameters = {
        'n_estimators': list(range(100, 200, 1)),
        'criterion': ('gini', 'entropy'),
    }
    
    
    CV = 5
    VERBOSE = 0
    grid_tuned = GridSearchCV(clf,
                          tuning_parameters,
                          cv=CV,
                          scoring='f1',
                          verbose=VERBOSE,
                          iid=True)
    grid_tuned.fit(X, y)
       
    return grid_tuned

def train_cnn(X, y):
    X = tf.expand_dims(X, axis=-1)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    model.fit(x=X, y=y)
    return model

def train_knn(X, y):
    clf = KNeighborsClassifier()
    tuning_parameters = {
        'n_neighbors': list(range(1, 10, 1)),
        'weights': ('uniform', 'distance'),
        'algorithm': ('ball_tree', 'kd_tree', 'brute')
    }
    
    CV = 5
    VERBOSE = 0
    grid_tuned = GridSearchCV(clf,
                          tuning_parameters,
                          cv=CV,
                          scoring='f1',
                          verbose=VERBOSE,
                          n_jobs=12,
                          iid=True)
    grid_tuned.fit(X, y)
        
    return grid_tuned

def train_SVC(X, y):
    clf = SVC()
    
    tuning_parameters = {
        'C': np.linspace(0.01, 2, num=50),
        'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
        'degree': list(range(2, 10, 1))
    }
    
    CV = 5
    VERBOSE = 0
    grid_tuned = GridSearchCV(clf,
                          tuning_parameters,
                          cv=CV,
                          scoring='f1_micro',
                          verbose=VERBOSE,
                          n_jobs=8,
                          iid=True)
    grid_tuned.fit(X, y)
        
    return grid_tuned
    
    # clf.fit(X, y)
    # return clf

#%% main
if __name__ == "__main__":
    X_train, y_train = load_data('chest_xray/chest_xray/train')
    X_test, y_test = load_data('chest_xray/chest_xray/test')
    X_val, y_val = load_data('chest_xray/chest_xray/val')


#%%
    shuffle_index = np.random.permutation(5216)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#%% Plots example image
    #cv2.imshow("image", X_train[0].reshape(img_size,img_size))

#%%
    #pca(64, X_train)
    
#%%
    start = time()
    model = train_cnn(X_train, y_train)
    t = time() - start
    X_test = tf.expand_dims(X_test, axis=-1)
    y_pred = model.predict(X_test)
    y_pred = np.rint(y_pred) # CNN gives out a procentage for how confident it is that it detects pheumonia this rounds that of to 1 or 0.
    # MSE = mean_squared_error(y_pred, y_test)
    # print(f'RMSE = {MSE**(1/2)}')
    print(f'F1-score: {f1_score(y_test, y_pred)}')
    print(f'accuracy: {accuracy_score(y_test, y_pred)}')
    print(confusion_matrix(y_test, y_pred))
    #b0, m0 = FullReport(model, X_test, y_test, t) # can and should be used if a grid- or random-seartch is done in the training function. 