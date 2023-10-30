import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy import stats
import pywt

## THIS FILE CONTAINS GENERAL FEATURES TO BE EXTRACTED FROM A WINDOW
def mav(window):
    ''' Extracts mean absolute value from window.'''
    return np.abs(window).mean(axis=0)

def wl(window):
    ''' Extracts waveform length number.'''
    return np.abs(np.diff(window, axis=0)).sum(axis=0)

def rms(window):
    ''' Extracts root mean square from window.'''
    return np.sqrt(np.square(window).mean(axis=0))

def skew(window):
    ''' Extracts skewness coefficient from given window.'''
    return np.nan_to_num(stats.skew(window, axis=0)) # extracts skew and sets nans to 0s

def iemg(window):
    ''' Extracts the integrated absolute EMG value.'''
    return np.abs(window).sum(axis=0)

def hjorth_params(window):
    ''' Extracts the activity, mobility and complexity Hjorth parameters.'''
    diff1 = np.diff(window, axis=0) # 1st derivative
    diff2 = np.diff(diff1, axis=0) # 2nd derivative
    activity, diff1_var, diff2_var = window.var(axis=0), diff1.var(axis=0), diff2.var(axis=0) # variance of signal and its 1st/2nd derivatives
    mobility = np.sqrt(diff1_var / activity) # mobility, measuring a sort of standard deviation of frequency
    complexity = np.sqrt(diff2_var / diff1_var) / mobility # complexity parameter, referring to a change in frequency

    # Treating results if any of the denominators are 0 (if signal is 0s, then so will the parameters)
    for ch in range(window.shape[1]):
        if np.isclose(activity[ch], 0): mobility[ch], complexity[ch] = 0, 0
        elif np.isclose(mobility[ch], 0): complexity[ch] = 0

    return list(activity) + list(mobility) + list(complexity) # return extracted features

def zero_crossings(window, ths):
    ''' Extracts zero-crossing feature based on specificically assigned threshold.'''
    pads = np.zeros((1, window.shape[1]))
    zcs = np.abs(np.diff(window > 0, axis=0, append=pads)) # number of times we get a zero crossing
    valid = np.abs(np.diff(window, axis=0, append=pads)) > ths
    return np.logical_and(zcs, valid).sum(axis=0)

# ADAPTED FROM COTE DALLARD 2018
def getAR(window, order=10):
    # Using Levinson Durbin prediction algorithm, get autoregressive coefficients
    # Square signal
    ARs = []
    for idx in range(window.shape[1]):
        vector = np.asarray(window[:, idx].flatten())
        R = [vector.dot(vector)]
        if R[0] == 0:
            ARs.append( np.array([1] + [0] * (order - 1) + [-1]) )
        else:
            for i in range(1, order + 1):
                r = vector[i:].dot(vector[:-i])
                R.append(r)
            R = np.array(R)
            # step 2:
            AR = np.array([1, -R[1] / R[0]])
            E = R[0] + R[1] * AR[1]
            for k in range(1, order):
                if (E == 0):
                    E = 10e-17
                alpha = - AR[:k + 1].dot(R[k + 1:0:-1]) / E
                AR = np.hstack([AR, 0])
                AR = AR + alpha * AR[::-1]
                E *= (1 - alpha ** 2)
            ARs.append(AR)
    return ARs

# ADAPTED FROM COTE DALLARD 2018
def getCC(window, order=4):
    CCs = []
    ARs = getAR(window, order=order)
    for idx in range(window.shape[1]):
        AR = ARs[idx]
        cc = np.zeros(order + 1)
        cc[0] = -1 * AR[0]  # issue with this line
        if order > 2:
            for p in range(2, order + 2):
                for l in range(1, p):
                    cc[p - 1] = cc[p - 1] + (AR[p - 1] * cc[p - 2] * (1 - (l / p)))
        CCs.extend(cc.tolist())
    return CCs

# Adapted from the Ulysse function from Cote Dallard
def mDWT_NinaPro(window, level=3, wavelet='db7'):
    '''Implements wavelet decomposition for feature extraction.'''
    Mxks = []
    for idx in range(window.shape[1]):
        coefficients = pywt.wavedec(window[idx], level=level, wavelet=wavelet)
        C = []
        for c in coefficients:
            C.extend(c)
        N = len(C)
        SMax = int(np.math.log(N, 2))
        Mxk = []
        for s in range(SMax):
            CMax = int(round((N / (2. ** (s + 1))) - 1))
            Mxk.append(np.sum(np.abs(C[0:(CMax)])))
        Mxks.extend(Mxk)
    return Mxks

# Adapted from Ulysse function from Cote Dallard
def hist(window, threshold_nmbr_of_sigma=3, bins=20):
    '''Computes histcounts for a given number of bins while accounting for outliers.'''
    threshold = threshold_nmbr_of_sigma
    hists = []
    for idx in range(window.shape[1]):
      hist, bin_edges = np.histogram(window, bins=bins, range=(-threshold, threshold))
      hists.extend(hist.tolist())
    return hists

def sampEn(window, m=1, r=0.2, tau=1):
    ''' SampEn is a proxy measure for the complexity of a given time-series. Computed on Zhang and Zhou, 2012.'''
    dist = lambda x, y : np.linalg.norm(x - y, ord=2, axis=0) # to compute euclidian distances
    Bm, Am, acc = np.zeros(window.shape[1]), np.zeros(window.shape[1]), 0
    for t in range(0, window.shape[0]-m-1, tau):
        for k in range(t+1, window.shape[0]-m-1, tau):
            Bm += (dist(window[t:t+m], window[k:k+m]) < r).astype(float) # if equal given tolerance, add 1
            Am += (dist(window[t:t+m+1], window[k:k+m+1]) < r).astype(float) # if equal given tolerance, add 1
            acc += 1
    Bm, Am = Bm/acc, Am/acc # transform counts into probabilities
    sampen = -np.log(Am / Bm) # sample entropy initial calculation
    sampen[sampen == np.inf] = -1 # set infinite values to -1 so we can reassign the valid maximum number to it
    sampen = np.nan_to_num(sampen, posinf=sampen.max())
    return sampen

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