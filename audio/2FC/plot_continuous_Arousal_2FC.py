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
dir_path = '/home/minhdanh/Documents/2FC_Audio/PredictedValues/Arousal/'    # CHECK
parser = argparse.ArgumentParser()
args = parser.parse_args()

# Call the binary files
movlist = ['BMI', 'LOR', 'GLA', 'DEP', 'CRA', 'CHI', 'FNE', 'ABE', 'MDB', 'NCO', 'RAT', 'SIL']

# Initial values
ave_maeArousal = 0.0
ave_mseArousal = 0.0
ave_pearsonArousal = 0.0
#

for index in range(0, len(movlist)):
    # ground truth values of arousal
    afilename_GT = 'my_continuous_arousal_for_only_audio.h5'       # CHECk
    h5file = h5py.File(os.path.join(dir_path, afilename_GT), mode = 'r')
    all_arousal = {}
    for k, v in h5file.items():
        all_arousal[int(k)] = v.value
    h5file.close()


    vfilename = movlist[index] + '_predArousal_emobase2010_2FC_Audio.h5'     # CHECK
    # DEP_predValence_emobase2010_2FC_classification_audio_video.h5
    h5file = h5py.File(os.path.join(dir_path, vfilename), mode = 'r')
    predArousal = h5file['default'].value
    #
    # Low pass filter
    # Calibrate the number of windows in Savitzky- Golay filter to get the best results
    # FOR AROUSAL:
    # 1701  for model with features from RGB
    # 1201 for model with features from RGB and OF.
    # 601 for model with RGB, OF and Audio(emobase2010)
    # 901 for model with RGB, OF
    # 901 for model with Audio(chroma_mfcc_prosody)
    # 901 for model with RGB, OF and Audio(chroma_mfcc_prosody)

    predArousal = butter_lowpass_filter(predArousal, cutoff, fs, order)
    predArousal = savgol_filter(predArousal, 701, 0, mode='interp')        # CHECK
    # Rescale predAroural to [-1, 1]
    predArousal = (predArousal.astype(float) - 3.0)/3.0
    h5file.close()
    #
    movlen = len(predArousal)
    x = pd.Series(range(0, movlen))
    #
    ypredArousal = pd.Series(predArousal)
    del predArousal
    torch.cuda.empty_cache()
    gc.collect()
    #

    yArousal = all_arousal[index][0:movlen]
    del all_arousal[index]
    torch.cuda.empty_cache()
    gc.collect()
    yArousal = pd.Series(yArousal)
    #
    # Apply median filter and plot results in seconds
    x = x/25
    yArousal = medfilt(yArousal)
    ypredArousal = medfilt(ypredArousal)
    #
    # Statistics:
    maeArousal = metrics.mean_absolute_error(yArousal, ypredArousal)
    mseArousal = metrics.mean_squared_error(yArousal, ypredArousal)
    pearsonArousal,_ = pearsonr(yArousal, ypredArousal)
    # Print statistical parameters:
    print('For: ', movlist[index])
    print('- Arousal: MAE: {:.5f}, MSE: {:.5f}, Pearson: {:.5f} \n'.format(maeArousal, mseArousal, pearsonArousal))

    #
    ave_maeArousal += maeArousal
    ave_mseArousal += mseArousal
    ave_pearsonArousal += pearsonArousal

    #==================================================================================================================
    # Data
    df = pd.DataFrame({'time': x,  'groundtruth_arousal': yArousal, 'predicted_arousal': ypredArousal})

    plt.plot( 'time', 'groundtruth_arousal', data=df, marker='', color='red', linewidth=1, label='groundtruth_arousal_'+str(movlist[index]))
    plt.plot( 'time', 'predicted_arousal', data=df, marker='', color='blue', linewidth=1, linestyle='dashed')
    plt.legend()
    #plt.show()


# For cross-validation
print('For leave-one-out cross-validation')
print('- Arousal: MAE: {:.5f}, MSE: {:.5f}, Pearson: {:.5f} \n'.format(ave_maeArousal/len(movlist), ave_mseArousal/len(movlist), ave_pearsonArousal/len(movlist)))


