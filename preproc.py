
import numpy as np
import scipy
import matplotlib.pyplot as plt
# from tqdm import tqdm

## PREPROCESSING

def standardize(EMG, getstats=False, means=None, stds=None):
    ''' Assumes that every column is the signal of a new channel.'''
    if means is None:
        means = EMG.mean(axis=0) # get current signals mean vector
    if stds is None:
        stds = EMG.std(axis=0) # get current signals std vector
    EMG_proc = EMG - means # center signal
    valid_ch = ~np.isclose(stds, 0)
    EMG_proc[:, ~valid_ch] = 0 # all dead channels set to 0 to prevent numerical errors
    EMG_proc[:, valid_ch] = EMG_proc[:, valid_ch] / stds[valid_ch] # standardize remaining channels
    if getstats:
        return EMG_proc, means, stds
    else:
        return EMG_proc

def maxmin(EMG):
    ''' Carries out max-min scaling of the EMG signals, assuming every column is a new channel.'''
    EMG_min = EMG.min(axis=0)
    EMG_max = EMG.max(axis=0)
    valid_ch = ~np.isclose(EMG_min, EMG_max) # if max=min, then channels are constant/dead
    EMG[:, ~valid_ch] = 0 # if dead channels, set to 0 to prevent numerical errors
    EMG[:, valid_ch] = (EMG[:, valid_ch] - EMG_min[valid_ch]) / (EMG_max[valid_ch]- EMG_min[valid_ch])
    return EMG

def notch(data):
    '''Used to remove powerline interference'''
    b, a = scipy.signal.iirnotch(50, Q=5, fs=2000) # creating notch filter parameters
    data = scipy.signal.filtfilt(b, a, data, axis=0) # applying notch filter to the data
    return data

def comb(data):
    '''Used to remove powerline interference and its multiples.'''
    b, a = scipy.signal.iircomb(50, Q=5, fs=2000) # creating comb filter params
    data = scipy.signal.filtfilt(b, a, data.T).T
    return data

def bandstop(data):
    '''Used to remove powerline interference and its multiples.'''
    sos = scipy.signal.butter(2, (45, 55), btype='bandstop', output='sos', fs=2000)
    data = scipy.signal.sosfilt(sos, data, axis=0)
    return data

def bandpass(data):
    '''Used to remove powerline interference and its multiples.'''
    sos = scipy.signal.butter(4, (20, 500), btype='bandpass', output='sos', fs=2000)
    data = scipy.signal.sosfilt(sos, data, axis=0)
    return data

def hampel(EMG, thrs=0.05, wlen=75):
    ''' 
        Takes in EMG segment and applies the Hampel outlier removal filter, replacing
        outliers with the median value of the given window.
    '''
    EMG_filt = np.copy(EMG) # array that will have removed outliers
    padding = np.zeros((wlen // 2, EMG.shape[1])) # padding to prepend to EMG
    EMG = np.vstack((padding, EMG))

    for t in tqdm(range(EMG_filt.shape[0] - wlen//2)): # Outlier detection and replacement
        vals = EMG[t, :] # values to undergo outlier detection
        win = EMG[t:t+wlen, :] # window to use to detect outlier
        med, mad = np.median(win, axis=0), scipy.stats.median_abs_deviation(win, axis=0)
        outliers = np.abs(vals - med) > thrs*mad # determine which channels contain outliers if any
        EMG_filt[t, outliers] = med[outliers] # assign median value to replace outliers
    return EMG_filt