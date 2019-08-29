#
# Audio features are extracted using emobase2010_1582_features.conf
#
import numpy as np
import os
import gc
import pandas as pd
import h5py
import time
from sklearn.preprocessing import MinMaxScaler

def standardized(given_data):
    # Standardize
    scaler = MinMaxScaler()
    scaler.fit(given_data)
    standardized_audio_feature_movie = scaler.transform(given_data)
    return standardized_audio_feature_movie

def main():
    start_time = time.time()
    pathIn = '/home/ubuntu/Documents/COGNIMUSE/OpenSmile'
    pathOut = '/home/ubuntu/Documents/COGNIMUSE/OpenSmile'
    movlist = ['BMI', 'LOR', 'GLA', 'DEP', 'CRA', 'CHI', 'FNE', 'ABE', 'MDB', 'NCO', 'RAT', 'SIL']

    standardized_aud_feature_list = {}
    for index in range(0, len(movlist)):
        print(movlist[index])

        filename = movlist[index] + '_aud_features.csv'
        filepath = os.path.join(pathIn, filename)

        with open(filepath, 'r') as csvFile:
            myDataFrame = pd.read_csv(csvFile, error_bad_lines=False, header = None, skiprows=1)   # remove the header, which is feature titles
            aud_feature = np.array(myDataFrame.iloc[:, 1: ], dtype=np.float32)  # remove the first column, which is the frame index

        standardized_aud_feature_list[index] = standardized(aud_feature[0:45175,:])
        csvFile.close()

        # Clear memory
	    #del myDataFrame
        #gc.collect()

    # Save extracted audio features into a .h5 file
    h5file = h5py.File(os.path.join(pathOut, 'audio_features_emobase2010_standardized.h5'), mode='w')
    for k, v in standardized_aud_feature_list.items():
        h5file.create_dataset(str(k), data=v, dtype=np.float32)   # str(k)
    h5file.close()

    print('Running time: ', time.time()-start_time)

if __name__ == "__main__":
    main()

