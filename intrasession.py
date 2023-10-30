import numpy as np
import scipy
import pandas as pd
import os
from copy import deepcopy
import pickle
from preproc import standardize
from manual_features.feature_sets import TD_features, ETD_features, NinaPro_features, SampEn_features
from manual_features.experiments import intrasession


## THIS FILE CONTAINS CODE FOR TAKING THE INTRASESSION EXPERIMENT MODULE AND RUNNING IT.

if __name__ == '__main__':

    DIR = '../datasets/ninapro/db2_subs'
    columns = ['Subject', 'Classifier', 'Train', 'Test', 'Feature Set', 'PCA']
    data = [] # to keep track of data entries for each aspect of the experiment
    # fset = TD_features
    for idx, fset in enumerate([TD_features]):#, ETD_features, NinaPro_features, SampEn_features]):
        fset_name = ['TD', 'ETD', 'NinaPro', 'SampEn'][idx]
        print('Feature Set #{}: {}'.format(idx+1, fset_name))

        for sub in range(1):
            row = [sub] # begin filling row
            print('Subject #{}'.format(sub+1))

            # Load given subject object
            file = open(os.path.join(DIR, 'sub_{}.pkl'.format(sub + 1)), 'rb')
            sub = pickle.load(file)
            file.close()

            # Split train/test data by extracting individual segments
            EMG, labels, indices = sub.emg, sub.labels, sub.indices
            EMG = standardize(EMG)

            rows = [deepcopy(row) for _ in range(8)] # one duplicate for each classifiers with and without PCA

            # Run Experiment
            train_accs, test_accs = intrasession(EMG, labels, indices, fset=fset, zscore=True, pca=True)
            for jdx, pca in enumerate(['False', 'True']):
                for kdx, clf_name in enumerate(['LDA', 'RF', 'KNN', 'SVM']):
                    rows[kdx + jdx*4].extend([clf_name, train_accs[kdx + jdx*4], test_accs[kdx + jdx*4]])
                    rows[kdx + jdx*4].append(fset_name)
                    rows[kdx + jdx*4].append(pca)
                
            data.extend(rows) # get rows from the current experiment condition
    
    # Creating dataframe and saving it
    df_res = pd.DataFrame(data=data, columns=columns)
    df_res.to_csv('./metrics.csv')


    # # Plotting confusion matrix
    # cm = confusion_matrix(y_test, y_test_pred, labels=mdl.classes_)
    # cm = cm / cm.sum(axis=0) # normalized confusion matrix
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm,
    #                                 display_labels=mdl.classes_)
    # disp.plot()
    # plt.title(mdl_name)
    # plt.show()

    # # Plotting prediction 'stream'
    # plt.figure()
    # plt.plot(y_test)
    # plt.plot(y_test_pred)
    # plt.legend(['Labels', 'Predictions'])
    # plt.show()


