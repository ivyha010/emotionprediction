import os
import argparse
import numpy as np
import gc
import torch as torch
import pandas as pd
import matplotlib.pyplot as plt
import time
import h5py
from scipy.signal import butter, lfilter, freqz, savgol_filter, medfilt, quadratic
from scipy.stats import pearsonr
from sklearn import metrics

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Filter requirements.
order = 6
fs =   30.0         # sample rate, Hz
cutoff = 3.667      # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

#----------------------------------------
start_time = time.time()
dir_path = '/home/minhdanh/Documents/2FC_Audio/PredictedValues/Valence/'    # CHECK
parser = argparse.ArgumentParser()
args = parser.parse_args()

# Call the binary files
movlist = ['BMI', 'LOR', 'GLA', 'DEP', 'CRA', 'CHI', 'FNE', 'ABE', 'MDB', 'NCO', 'RAT', 'SIL']

# Initial values
ave_maeValence = 0.0
ave_mseValence = 0.0
ave_pearsonValence = 0.0
#

for index in range(0, len(movlist)):
    # ground truth values of valence
    afilename_GT = 'my_continuous_valence_for_only_audio.h5'       # CHECk
    h5file = h5py.File(os.path.join(dir_path, afilename_GT), mode = 'r')
    all_valence = {}
    for k, v in h5file.items():
        all_valence[int(k)] = v.value
    h5file.close()


    vfilename = movlist[index] + '_predValence_emobase2010_2FC_Audio.h5'     # CHECK
    # DEP_predValence_emobase2010_2FC_classification_audio_video.h5
    h5file = h5py.File(os.path.join(dir_path, vfilename), mode = 'r')
    predValence = h5file['default'].value
    #
    # Low pass filter
    # Calibrate the number of windows in Savitzky- Golay filter to get the best results
    # FOR VALENCE:
    # 1701  for model with features from RGB
    # 1201 for model with features from RGB and OF.
    # 601 for model with RGB, OF and Audio(emobase2010)
    # 901 for model with RGB, OF
    # 901 for model with Audio(chroma_mfcc_prosody)
    # 901 for model with RGB, OF and Audio(chroma_mfcc_prosody)

    predValence = butter_lowpass_filter(predValence, cutoff, fs, order)
    predValence = savgol_filter(predValence, 1321, 0, mode='interp')        # CHECK
    # Rescale predAroural to [-1, 1]
    predValence = (predValence.astype(float) - 3.0)/3.0
    h5file.close()
    #
    movlen = len(predValence)
    x = pd.Series(range(0, movlen))
    #
    ypredValence = pd.Series(predValence)
    del predValence
    torch.cuda.empty_cache()
    gc.collect()
    #

    yValence = all_valence[index][0:movlen]
    del all_valence[index]
    torch.cuda.empty_cache()
    gc.collect()
    yValence = pd.Series(yValence)
    #
    # Apply median filter and plot results in seconds
    x = x/25
    yValence = medfilt(yValence)
    ypredValence = medfilt(ypredValence)
    #
    # Statistics:
    maeValence = metrics.mean_absolute_error(yValence, ypredValence)
    mseValence = metrics.mean_squared_error(yValence, ypredValence)
    pearsonValence,_ = pearsonr(yValence, ypredValence)
    # Print statistical parameters:
    print('For: ', movlist[index])
    print('- Valence: MAE: {:.5f}, MSE: {:.5f}, Pearson: {:.5f} \n'.format(maeValence, mseValence, pearsonValence))

    #
    ave_maeValence += maeValence
    ave_mseValence += mseValence
    ave_pearsonValence += pearsonValence

    #==================================================================================================================
    # Data
    df = pd.DataFrame({'time': x,  'groundtruth_valence': yValence, 'predicted_valence': ypredValence})

    plt.plot( 'time', 'groundtruth_valence', data=df, marker='', color='red', linewidth=1, label='groundtruth_valence_'+str(movlist[index]))
    plt.plot( 'time', 'predicted_valence', data=df, marker='', color='blue', linewidth=1, linestyle='dashed')
    plt.legend()
    #plt.show()


# For cross-validation
print('For leave-one-out cross-validation')
print('- Valence: MAE: {:.5f}, MSE: {:.5f}, Pearson: {:.5f} \n'.format(ave_maeValence/len(movlist), ave_mseValence/len(movlist), ave_pearsonValence/len(movlist)))


