
from scipy.ndimage import filters
import numpy as np
import random
import string
import copy
import pywt
from scipy import interpolate
from scipy import signal, ndimage
from astropy.convolution import Gaussian1DKernel,convolve

def preprocessing(feature, selection=None):
    feature = np.array(feature)

    if selection == 'wl':
        wt = pywt.wavedec(feature, 'db1', level=4)
        y_wave = wt[selection]
        wavelen =  len(y_wave)
        x_wave = np.linspace(0, wavelen-1, wavelen)
        y_inter = interpolate.interp1d(x_wave,y_wave)
        x_extend = np.linspace(0, wavelen-1, 2600)
        feature = y_inter(x_extend)
    elif selection == 'lp':
        b, a = signal.butter(2, 0.3, 'low')
        feature = signal.filtfilt(b, a, feature)
    elif selection == 'gs':
        y_g = signal.savgol_filter(feature, 5, 2)
        g = Gaussian1DKernel(stddev=3)
        feature = convolve(y_g, g)

    return feature.tolist()
