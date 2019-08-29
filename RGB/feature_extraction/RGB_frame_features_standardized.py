import h5py
import os
import torch
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler

def read_h5py_files(filename, pathIn, pathOut):      # num_PCs: number of principle components

    start_time = time.time()

    # Read features from h5py file
    standardized_OF_feature_list = {}
    h5file = h5py.File(os.path.join(pathIn,filename), 'r')
    for k, v in h5file.items():
        OF_features_movie = torch.from_numpy(v.value)

        # Standardize
        scaler = MinMaxScaler()
        scaler.fit(OF_features_movie)
        standardized_OF_feature_list[int(k)] = scaler.transform(OF_features_movie)

    h5file.close()

    # Save in a h5py file
    h5file = h5py.File(os.path.join(pathOut, 'RGB_features_ResNet50_standardized_new.h5'), mode='w')
    for k, v in standardized_OF_feature_list.items():
        h5file.create_dataset(str(k), data=v, dtype=np.float32)
    h5file.close()

    print('Running time: ', time.time() - start_time)


def main():
    path_in = '/home/ubuntu/Documents/COGNIMUSE/'
    path_out = '/home/ubuntu/Documents/COGNIMUSE/'
    file_name = 'ResNet50_RGB_new.h5'
    read_h5py_files(file_name, path_in, path_out)


if __name__ == "__main__":
    main()
