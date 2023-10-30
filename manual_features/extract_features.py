## THIS FILE CONTAINS UTILITY FUNCTIONS FOR TRADITIONAL FEATURE EXTRACTION
import numpy as np
import scipy
import matplotlib.pyplot as plt
from feature_sets import TD_features
from tqdm import tqdm

def extract_features(EMG, labels, fset, stride=0.015, wlen=0.1, fs=2000):
    ''' Takes in the object containing EMG data and labels and extracts features based
        on the selected fset. For now, assigns to a window the mode of all labels in the given window.
        
        Parameters:
        EMG (ndarray): A TxCh matrix containing all channels of EMG for a given segment.
        labels (ndarray): A Tx1 column vector containing stream of labels for EMG segment.
        fset (function): A function that take in an EMG window, and returns feature vector.
        stride (int): Stride time used for extracting windows from EMG signal.
        wlen (int): Length of time-windows to extract features from.
        fs (int): Sampling rate.

        Returns:
        X (ndarray): A feature matrix of size window index by number of features (size of feature set
            times number of channels).
        y (ndarray): A column vector containing one label per time-window.
    '''
    # Get number of features in feature set
    rand_wind = np.random.normal(size=(200, 1)) # random window to get feature set info
    fset_size = len(fset(rand_wind)) # get number of features outputted

    # Setup data containers and appropriate parameters
    stride_samp, wlen_samp = int(fs*stride), int(fs*wlen) # get vars in terms of samples
    Nw = (EMG.shape[0] - wlen_samp) // stride_samp + 1 # number of windows to extract
    X = np.zeros((Nw, EMG.shape[1]*fset_size)) # initialize feature matrix
    y = np.zeros((Nw, 1)) # initialize label vector

    # Get features from each window
    for idx, widx in enumerate(range(0, EMG.shape[0] - wlen_samp, stride_samp)):
        win = EMG[widx:widx+wlen_samp, :] # get time-window of interest
        X[idx, :] = np.array(fset(win)) # extract relevant features
        y[idx] = scipy.stats.mode(labels[widx:widx+wlen_samp].ravel(), keepdims=False)[0] # get label to be assigned to given time-window
    return X, y
 
if __name__ == '__main__':
    EMG = np.random.normal(size=(100000, 12)) # random EMG signal
    labels = np.array([[int(idx % 2 == 0)]*10000 for idx in range(10)]).reshape(-1, 1) # labels
    X, y = extract_features(EMG, labels, fset=TD_features) # Extract some features boai
    plt.figure()
    plt.plot(X[:,1])
    plt.plot(y.ravel())
    plt.legend(['EMG', 'Label'])
    plt.show()

