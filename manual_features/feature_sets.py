import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy import stats
import pywt
from features import mav, zero_crossings, wl, rms, skew, iemg, getAR, hjorth_params,\
    mDWT_NinaPro, hist, sampEn, getCC


## FEATURE SETS IMPLEMENTED
def TD_features(window, ths=None):
    ''' Takes in a particular window, and returns row containing all the manually extracted features for all channels from Hudgins and Ecklehart., 2003'''
    if ths == None: ths = np.zeros((1,window.shape[1]))
    features = [] # initialize row
    features.extend(list(mav(window))) # add mean absolute value
    features.extend(list(zero_crossings(window, ths)))
    features.extend(list(np.abs(np.diff(np.diff(window, axis=0) > 0, axis=0)).sum(axis=0))) # add number of slope changes
    features.extend(list(wl(window))) # adding the waveform length number
    return features

def ETD_features(window):
    ''' Takes in a particular window, and returns row containing all the manually extracted features for all channels from Khushaba et al., 2012'''
    features = [] # initialize row
    features.extend(TD_features(window)) # original time-domain features
    features.extend(list(rms(window))) # root mean square feature
    features.extend(list(skew(window))) # skewness feature
    features.extend(list(iemg(window))) # integrated absolute emg feature
    features.extend([a for coeffs in getAR(window) for a in coeffs])
    features.extend(hjorth_params(window)) # get activity, mobility and coplexity hjorth parameters
    return features

def NinaPro_features(window):
    ''' Takes in a particular window, and returns row containing all the manually extracted features for all channels from Atzori et al., 2014'''
    features = [] # initialize row
    features.extend(TD_features(window))
    features.extend(list(rms(window))) # root mean square feature
    features.extend(list(mDWT_NinaPro(window))) # wavelet decomposition features
    features.extend(list(hist(window))) # extracts histogram based features
    return features

def SampEn_features(window):
    '''Takes in a particular window, and returns row containing all manually extracted features determined most stable by Phinoyamark et al., 2012'''
    features = [] # initialize row
    features.extend(list(sampEn(window))) # SampEn estimation
    features.extend(getCC(window)) # 4th order Cepstral coefficients
    features.extend(list(rms(window))) # root-mean square
    features.extend(list(wl(window))) # waveform length number 
    return features