import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import pickle
from utility.generate_db import NinaProSubject, create_db
from manual_features.extract_features import extract_features
from feature_sets import TD_features, ETD_features, NinaPro_features, SampEn_features
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score
from tqdm import tqdm
from preproc import standardize

def intrasession(EMG, labels, indices, fset=TD_features, zscore=False, pca=False):
    ''' Intrasession experiment for given EMG and labels. Returns train/test accuracies of each model.
        Inputs:
            EMG[ndarray]: TxNch array containing EMG signal for the given session
            labels[ndarray]: Tx1 array containing label signal for given session
            indices[ndarray]: contains the start and end indices of each gesture
            fset[func]: feature set function that extracts F.Nch features from a given window
            zscore[bool]: whether to apply z-score normalization to features extracted
            pca[bool]: whether to also obtain results with PCA dimensionality reduction
                (>95% explained variance) in the output
        Returns:
            train_accs[ndarray]: either (4,) or (8,) shaped array, containing train accuracy
                for each classifier, and potentially with/without PCA dimesionality reduction
            test_accs[ndarray]: same as above, but the test accuracy
            bal_test_accs[ndarray]: same as above, but the accuracy weighted by class (averaged
                recall of each class)
    '''

    # Split train/test data by extracting individual segments
    print('Extracting Features...')
    prev_end = 0 # initializing the idx of the previous gesture location
    EMG_train, labels_train, EMG_test, labels_test = [], [], [], []

    # Only get indices of gesture ends
    for idx, cur_end in enumerate(indices[1::2]): 
        if idx % 2 == 0: # every other gesture
            EMG_train.append(EMG[prev_end:cur_end, :])
            labels_train.append(labels[prev_end:cur_end])
        else:
            EMG_test.append(EMG[prev_end:cur_end, :])
            labels_test.append(labels[prev_end:cur_end])
        prev_end = cur_end
    assert len(EMG_test) == len(EMG_train), 'Number of gestures in train set must match number of gestures in test set.'
    
    # Construct feature/label matrices based on assigned segments
    rand_wind = np.random.normal(size=(200, 1)) # random window to get feature set info
    fset_size = len(fset(rand_wind)) # get number of features outputted
    X_train, X_test = np.zeros((0, fset_size*EMG.shape[1])), np.zeros((0, fset_size*EMG.shape[1]))
    y_train, y_test = np.zeros((0, 1)), np.zeros((0, 1))
    for idx in tqdm(range(len(EMG_train))):
        # Extract features and appropriate labels from segment
        X_train_seg, y_train_seg = extract_features(EMG_train[idx], labels_train[idx], fset=fset)
        X_test_seg, y_test_seg = extract_features(EMG_test[idx], labels_test[idx], fset=fset)
        # Adding to feature/label matrices
        X_train, X_test = np.vstack((X_train, X_train_seg)), np.vstack((X_test, X_test_seg))
        y_train, y_test = np.vstack((y_train, y_train_seg)), np.vstack((y_test, y_test_seg))

    # Sklearn preferred labels shape
    y_train, y_test = y_train.ravel(), y_test.ravel()

    # Handling training label imbalance
    X_train, y_train = resample(X_train, y_train) # upsamples minority classes

    # Feature preprocessing
    if zscore: 
        X_train, means, stds = standardize(X_train, getstats=True)
        X_test = standardize(X_test, means=means, stds=stds)
    if pca:
        pca = PCA(n_components=0.95).fit(X_train) # components for 95% of explained variance from the training set
        X_train_pca = pca.transform(X_train) # PCA'd training features
        X_test_pca = pca.transform(X_test) # PCA'd testing features

    # Training/testing
    print('Training classifiers...')
    mdls = [LinearDiscriminantAnalysis().fit(X_train, y_train), 
            RandomForestClassifier().fit(X_train, y_train), 
            KNeighborsClassifier().fit(X_train, y_train),
            SVC().fit(X_train, y_train)]
    
    # Obtaining test performance
    print('Evaluating performances...')
    train_accs, test_accs, bal_test_accs = [], [], []
    for mdl_name, mdl in zip(['LDA', 'RF', 'KNN', 'SVM'], mdls):
        y_train_pred, y_test_pred = mdl.predict(X_train), mdl.predict(X_test)
        train_acc, test_acc = accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)
        bal_test_acc = balanced_accuracy_score(y_test, y_test_pred)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        bal_test_accs.append(bal_test_acc)
        print('MDL: ' + mdl_name)
        print('Train Performance: {}, Test Performance: {}, Balanced Test Performance: {} \n'.format(train_acc, test_acc, bal_test_acc))

    # If PCA is set to True, also compute performances with PCA
    if pca:
        pca = PCA(n_components=0.95).fit(X_train) # components for 95% of explained variance from the training set
        X_train_pca = pca.transform(X_train) # PCA'd training features
        X_test_pca = pca.transform(X_test) # PCA'd testing features
        print('Training PCA classifiers')
        mdls_pca = [LinearDiscriminantAnalysis().fit(X_train_pca, y_train), 
            RandomForestClassifier().fit(X_train_pca, y_train), 
            KNeighborsClassifier().fit(X_train_pca, y_train),
            SVC().fit(X_train_pca, y_train)]
        
        print('Evaluating performances with PCA...')
        for mdl_name, mdl in zip(['LDA', 'RF', 'KNN', 'SVM'], mdls_pca):
            y_train_pred, y_test_pred = mdl.predict(X_train_pca), mdl.predict(X_test_pca)
            train_acc, test_acc = accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)
            bal_test_acc = balanced_accuracy_score(y_test, y_test_pred)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            bal_test_accs.append(bal_test_acc)
            print('MDL: ' + mdl_name)
            print('Train Performance: {}, Test Performance: {}, Balanced Test Performance: {} \n'.format(train_acc, test_acc, bal_test_acc))

    return train_accs, test_accs


## INTERSUBJECT OR INTERSESSION!
def interfile(EMG_train, labels_train, EMG_test, labels_test, fset=TD_features, zscore=False, pca=False):
    ''' Intersubject/intersession experiment for given EMG and labels that were obtained from different 
        files. Returns train/test accuracies of each model.
        Inputs:
            EMG_train[ndarray]: TxNch array containing EMG signal for the given train session
            labels_train[ndarray]: Tx1 array containing label signal for given train session
            EMG_test[ndarray]: TxNch array containing EMG signal for the given test session
            labels_test[ndarray]: Tx1 array containing label signal for given test session
            fset[func]: feature set function that extracts F.Nch features from a given window
            zscore[bool]: whether to apply z-score normalization to features extracted
            pca[bool]: whether to also obtain results with PCA dimensionality reduction
                (>95% explained variance) in the output
        Returns:
            train_accs[ndarray]: either (4,) or (8,) shaped array, containing train accuracy
                for each classifier, and potentially with/without PCA dimesionality reduction
            test_accs[ndarray]: same as above, but the test accuracy
            bal_test_accs[ndarray]: same as above, but the accuracy weighted by class (averaged
                recall of each class)
    '''
    # Train on signals from one session and test on another
    print('Extracting Features...')

    # Construct feature/label matrices based on assigned segments
    X_train, y_train = extract_features(EMG_train, labels_train, fset=fset)
    X_test, y_test = extract_features(EMG_test, labels_test, fset=fset)

    # Sklearn preferred labels shape
    y_train, y_test = y_train.ravel(), y_test.ravel()

    # Handling training label imbalance
    X_train, y_train = resample(X_train, y_train) # upsamples minority classes

    # Feature preprocessing
    if zscore: 
        X_train, means, stds = standardize(X_train, getstats=True)
        X_test = standardize(X_test, means=means, stds=stds)
    if pca:
        pca = PCA(n_components=0.95).fit(X_train) # components for 95% of explained variance from the training set
        X_train_pca = pca.transform(X_train) # PCA'd training features
        X_test_pca = pca.transform(X_test) # PCA'd testing features

    # Training/testing
    print('Training classifiers...')
    mdls = [LinearDiscriminantAnalysis().fit(X_train, y_train), 
            RandomForestClassifier().fit(X_train, y_train), 
            KNeighborsClassifier().fit(X_train, y_train),
            SVC(max_iter=1).fit(X_train, y_train)]
    
    # Obtaining test performance
    print('Evaluating performances...')
    train_accs, test_accs, bal_test_accs = [], [], []
    for mdl_name, mdl in zip(['LDA', 'RF', 'KNN', 'SVM'], mdls):
        y_train_pred, y_test_pred = mdl.predict(X_train), mdl.predict(X_test)
        train_acc, test_acc = accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)
        bal_test_acc = balanced_accuracy_score(y_test, y_test_pred)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        bal_test_accs.append(bal_test_acc)
        print('MDL: ' + mdl_name)
        print('Train Performance: {}, Test Performance: {}, Balanced Test Performance: {} \n'.format(train_acc, test_acc, bal_test_acc))

    # If PCA is set to True, also compute performances with PCA
    if pca:
        pca = PCA(n_components=0.95).fit(X_train) # components for 95% of explained variance from the training set
        X_train_pca = pca.transform(X_train) # PCA'd training features
        X_test_pca = pca.transform(X_test) # PCA'd testing features
        print('Training PCA classifiers')
        mdls_pca = [LinearDiscriminantAnalysis().fit(X_train_pca, y_train), 
            RandomForestClassifier().fit(X_train_pca, y_train), 
            KNeighborsClassifier().fit(X_train_pca, y_train),
            SVC().fit(X_train_pca, y_train)]
        
        print('Evaluating performances with PCA...')
        for mdl_name, mdl in zip(['LDA', 'RF', 'KNN', 'SVM'], mdls_pca):
            y_train_pred, y_test_pred = mdl.predict(X_train_pca), mdl.predict(X_test_pca)
            train_acc, test_acc = accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)
            bal_test_acc = balanced_accuracy_score(y_test, y_test_pred)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            bal_test_accs.append(bal_test_acc)
            print('MDL: ' + mdl_name)
            print('Train Performance: {}, Test Performance: {}, Balanced Test Performance: {} \n'.format(train_acc, test_acc, bal_test_acc))

    return train_accs, test_accs, bal_test_accs