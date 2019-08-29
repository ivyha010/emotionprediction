import pickle
import subprocess
import numpy as np
import argparse
import torch
import torch.utils.data as data
from torchvision import transforms
from data_loader import CognimuseDataset
from pretrainedCNNs import FeatureExtraction_ResNet18, FeatureExtraction_ResNet34, FeatureExtraction_ResNet50, \
    FeatureExtraction_ResNet101, FeatureExtraction_ResNet152, FeatureExtraction_VGG19, FeatureExtraction_VGG16, FeatureExtraction_AlexNet
import os
import time
import gc
from collections import Mapping, Container
from sys import getsizeof
import h5py

def deep_getsizeof(o, ids):
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, np.unicode):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r

# Memory check
def memoryCheck():
    ps = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv'], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    print(ps.communicate(), '\n')
    os.system("free -m")

# Free memory
def freeCacheMemory():
    torch.cuda.empty_cache()
    gc.collect()

# Extract features from all movies
def extractFeature(movlist, featureExtMethod_RGB, img_dataloaders, device):
    movlistlength = len(movlist)
    img_features_list = {}  # list of features from all movies
    valence_list = {}
    arousal_list = {}
    #
    for index in range(0, movlistlength):
        start_time = time.time()
        img_features = []   # features extracted from images from each movie
        valence_temp = []
        arousal_temp = []
        #
        count = 0
        print('Extracting features from ', movlist[index], '.....')
        for (images, valence, arousal) in img_dataloaders[index]:
            # Set mini-batch dataset (inputs and targets)
            images = images.to(device)
            valence, arousal = valence.to(device), arousal.to(device)
            valence_temp.append(valence)
            arousal_temp.append(arousal)
            #
            temp = featureExtMethod_RGB(images)
            # img_features.append(np.array(featureExtMethod_RGB(images).detach().numpy()))
            img_features.append(featureExtMethod_RGB(images))

            if (count % 200) == 0:
                print('Batch: ', count)
            count += 1
            #
            # Free memory - Delete images
            del images
            freeCacheMemory()

        img_list_temp = torch.cat(img_features, dim=0).to('cpu')       # np.concatenate(img_features, axis=0)
        valence_list_temp = torch.cat(valence_temp, dim=0).to('cpu')   # np.concatenate(valence_temp, axis=0)
        arousal_list_temp = torch.cat(arousal_temp, dim=0).to('cpu')   # np.concatenate(arousal_temp, axis=0)

        del img_features, valence_temp, arousal_temp
        freeCacheMemory()
        #
        length = min(img_list_temp.shape[0], valence_list_temp.shape[0], arousal_list_temp.shape[0])

        img_features_list[index] = img_list_temp[0: length, :].clone()
        del img_list_temp
        freeCacheMemory()

        valence_list[index] = valence_list_temp[0: length].clone()
        arousal_list[index] = arousal_list_temp[0: length].clone()
        #
        del valence_list_temp, arousal_list_temp
        freeCacheMemory()

        print('Extracted features from: ', movlist[index], 'in ', time.time() - start_time, ' seconds')

    return img_features_list, valence_list, arousal_list

# Main
def main(args):
    # Device configuration
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Manual seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device: ', device)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Image preprocessing: data augmentation, normalization for the pretrained resnet
    Transform_RGB = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    # Call the dataset
    img_datasets = []
    movlistlength = len(movlist)
    for index in range(0, movlistlength):
        img_datasets.append(CognimuseDataset(img_csvfiles[index], img_folders[index], Transform_RGB))

    # Build data loader
    img_dataloaders = []
    for index in range(0, movlistlength):
        img_dataloaders.append(torch.utils.data.DataLoader(dataset=img_datasets[index], batch_size=args.batch_size,
                                                     shuffle=False, **kwargs))     # Note: shuffle=False

    memoryCheck()
    # Prepare for feature extraction
    if args.pretrained_CNN == 'ResNet18':
        feature_extraction_RGB = FeatureExtraction_ResNet18().to(device)
    elif args.pretrained_CNN == 'ResNet34':
        feature_extraction_RGB = FeatureExtraction_ResNet34().to(device)
    elif args.pretrained_CNN == 'ResNet50':
        feature_extraction_RGB = FeatureExtraction_ResNet50().to(device)
    elif args.pretrained_CNN == 'ResNet101':
        feature_extraction_RGB = FeatureExtraction_ResNet101().to(device)
    elif args.pretrained_CNN == 'ResNet152':
        feature_extraction_RGB = FeatureExtraction_ResNet152().to(device)
    elif args.pretrained_CNN == 'VGG16':
        feature_extraction_RGB = FeatureExtraction_VGG16().to(device)
    elif args.pretrained_CNN == 'VGG19':
        feature_extraction_RGB = FeatureExtraction_VGG19().to(device)
    elif args.pretrained_CNN == 'AlexNet':
        feature_extraction_RGB = FeatureExtraction_AlexNet().to(device)
    else:
        feature_extraction_RGB = 0
        print('Check the pretrained CNN model')

    # Use pretrained model
    feature_extraction_RGB.eval()

    # Extract features
    start_time = time.time()
    img_features_list, valence_list, arousal_list = extractFeature(movlist, feature_extraction_RGB, img_dataloaders, device)
    print('Running time for feature extraction for all movie clips:', time.time() - start_time, ' seconds')
    memoryCheck()

    # Save extracted features, valence list, arousal list
    h5file = h5py.File(os.path.join(dir_path_out, args.pretrained_CNN +'_RGB_new.h5'), mode='w')
    for k, v in img_features_list.items():
        v = v.cpu()
        h5file.create_dataset(str(k), data=v.detach().numpy(), dtype=np.float32)   # str(k)
    h5file.close()
    #
    h5file = h5py.File(os.path.join(dir_path_out, 'my_continuous_valence_file_RGB_new.h5'), mode='w')
    for k, v in valence_list.items():
        h5file.create_dataset(str(k), data=v, dtype=np.float32)   # str(k)
    h5file.close()
    #
    h5file = h5py.File(os.path.join(dir_path_out, 'my_continuous_arousal_file_RGB_new.h5'), mode='w')
    for k, v in arousal_list.items():
        h5file.create_dataset(str(k), data=v, dtype=np.float32)   # str(k)
    h5file.close()


if __name__ == "__main__":
    #
    dir_path_in = '/home/ubuntu/Documents/COGNIMUSE_Draft'  # '/home/ubuntu/Documents/CNN_LSTM/'
    dir_path_out = '/home/ubuntu/Documents/COGNIMUSE_Draft'  # '/home/ubuntu/Documents/COGNIMUSE/'
    # ------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=224, help='size for cropping images') 

    parser.add_argument('--pretrained_CNN', type = str, default = 'ResNet50', help = 'pretrained CNN model for feature extraction')  ResNet34, ResNet101, ResNet50, ResNet152, VGG16, VGG19, AlexNet

    parser.add_argument('--batch_size', type=int, default=25, help = 'number of frames used for feature extraction each time')  

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    args = parser.parse_args()
    print(args)

    # ------------------------------------------------------------------------------------------------------------------
    movlist = ['BMI', 'LOR', 'GLA', 'DEP', 'CRA', 'CHI', 'FNE', 'ABE', 'MDB', 'NCO', 'RAT', 'SIL']
    img_folders = []        # folders of images(video frames)
    img_csvfiles = []       # .csv files containing image names, valence and arousal values
    for movie in movlist:
        img_folders.append(dir_path_in + movie)                    # movie: movie's title
        img_csvfiles.append(dir_path_in + 'intended3_' + movie + '.csv') # 'ave_' + movie + '.csv':  csv file name

    #-------------------------------------------------------------------------------------------------------------------
    # Note: OF_image_names.csv and image-values.csv must have the same row numbers (number of opt. flow images = numb of images)
    main_start_time = time.time()
    main(args)
    print('Total running time: {:.5f} seconds' .format(time.time() - main_start_time))
