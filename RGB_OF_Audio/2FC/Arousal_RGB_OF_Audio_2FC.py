import subprocess
import numpy as np
import argparse
import torch
from torch import optim, nn
from two_FC_layer_model_RGB_OF_Audio import Two_FC_layer
import os
import time
import gc
from collections import Mapping, Container
from sys import getsizeof
import h5py
from torch.utils.data import DataLoader, Dataset
from pytorchtools import EarlyStopping
from scipy.stats import pearsonr
from sklearn import metrics
from torch.nn import functional as F


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

# Build dataloaders
def train_dataloader_for_FC_model_Arousal(trfeatures, trarousal, args):
    class my_dataset(Dataset):
        def __init__(self, data, label):
            self.data = data
            self.label = label

        def __getitem__(self, index):
            return self.data[index], self.label[index]

        def __len__(self):
            return len(self.data)

    # Convert a dictionary to a tensor
    train_features = np.concatenate([value.unsqueeze(0) for _, value in trfeatures.items()], axis=1)
    train_features = train_features.squeeze(0)
    #
    train_arousal = np.concatenate([value.unsqueeze(0) for _, value in trarousal.items()], axis=1)
    train_arousal = train_arousal.reshape(-1, 1)
    #
    # Build dataloaders
    train_loader = DataLoader(dataset=my_dataset(np.array(train_features), train_arousal), batch_size=args.batch_size, shuffle=True)
    #
    return train_loader

def validate_dataloader_for_FC_model_Arousal(tfeatures, tarousal, tarousal_cont, args):
    class my_dataset(Dataset):
        def __init__(self, data, label, cont_gtruth):
            self.data = data
            self.label = label
            self.cont_gtruth = cont_gtruth

        def __getitem__(self, index):
            return self.data[index], self.label[index], self.cont_gtruth[index]

        def __len__(self):
            return len(self.data)

    # Build dataloaders
    validate_loader = DataLoader(dataset=my_dataset(np.array(tfeatures), np.array(tarousal.reshape(-1,1)), np.array(tarousal_cont.reshape(-1,1))), batch_size=args.batch_size, shuffle=False)
    #
    return validate_loader

# Train
def train_func(train_loader, vfeature, varousal, the_model, device, criter, optimizer, n_epochs, input_size, patience):

    start_time = time.time()
    the_model.train()  # pre model for training
    #
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    #
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, n_epochs + 1):
        # Adjust learning rate
        # adjust_learning_rate(optimizer, epoch)
        ###################
        # train the model #
        ###################
        the_model.train()  # prep model for training

        for (feature, arousal) in train_loader:
            feature, arousal = feature.to(device), arousal.to(device)
            #
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = the_model.forward(feature.reshape(-1, input_size))
            output = output/T
            # calculate the loss
            # KL Loss
            # output = F.log_softmax(output, dim=1)
            # loss = criter(output.float(), arousal.float())
            #-----------------------------------------------------------------------------
            # Cross Entropy Loss
            loss = criter(output.squeeze(1), arousal.squeeze(1))  # CrossEntropy Loss
            #-----------------------------------------------------------------------------
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward(retain_graph=True)
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        the_model.eval()  # prep model for evaluation
        vfeature, varousal = vfeature.to(device), varousal.to(device)
        valid_output = the_model(vfeature)
        valid_output = valid_output/T

        # validation loss:
        # Cross Entropy Loss
        valid_loss = criter(valid_output.squeeze(1), varousal)
        #----------------------------------------------------------------------------
        # KL loss
        #valid_output = F.log_softmax(valid_output,dim=1)
        #valid_loss = criter(valid_output.float(), varousal.unsqueeze(1).float())
        #----------------------------------------------------------------------------
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)

        avg_valid_losses.append(valid_loss.item())
        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}]' +
                     f' train_loss: {train_loss:.8f} ' +
                     f' valid_loss: {valid_loss:.8f} ')
        print(print_msg)

        # clear lists to track next epoch
        train_losses = []

        # early_stopping needs the (1-valid_pearson) to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss.item(), the_model)

        print('Epoch[{}/{}]: Training time: {} seconds '.format(epoch,n_epochs, time.time() - start_time))
        start_time = time.time()

        #
        del valid_output
        freeCacheMemory()

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    the_model.load_state_dict(torch.load('checkpoint.pt'))

    return the_model, avg_train_losses, avg_valid_losses


# Validate
def validate_func(feature, arousal, the_model, device):
    #
    the_model.eval()
    #
    feature, arousal = feature.to(device), arousal.to(device)
    output = the_model(feature)
    output /= T

    # Accuracy and Accuracy +-1
    _, prediction = torch.max(output.data, 1)
    # prediction = prediction.cpu().numpy()
    test_acc = torch.sum(prediction == arousal)
    # Compute the average accuracy and loss over all validate dataset
    test_acc = np.float32(test_acc.item()/output.size()[0])

    test_acc_1 = 0
    bin_bias = np.abs((prediction - arousal).cpu())
    for element in bin_bias:
        if element.item() == 1:
            test_acc_1 += 1
    test_acc_1 = test_acc_1/output.size()[0]

    print('Validation (Use both Audio and Video features): ')
    print('- Discrete case: For Arousal: Accuracy: {:.5f} %, Accuracy+/-1: {:.5f} % \n'.format(100 * test_acc, 100 * test_acc_1))

    return prediction, test_acc, test_acc_1



# Decay the learning rate
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    newlr = args.lr * (0.1 ** (epoch // 25))
    for param_group in optimizer.param_groups:
        param_group['lr'] = newlr

# Checkpoint
def checkpoint(model_checkpoint, epoch):
    model_out_path = dir_path + 'Thao_model/' + "model_epoch_{}.pth".format(epoch)
    torch.save(model_checkpoint, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

# Load extracted features and arousal files
def loadingfiles(device):
    # Load extracted features and arousal .h5 files
    print('\n')
    print('Loading h5 files containing extracted features and arousal values.....')
    loading_time = time.time()
    h5file = h5py.File(os.path.join(dir_path, 'rgb_OF_audio_concat.h5'), 'r')
    train_features = {}
    for k, v in h5file.items():
        train_features[int(k)] = torch.from_numpy(v.value) #.to(device)  # Convert numpy arrays to tensors on gpu  # .to(device)
    h5file.close()
    #
    print('Time for loading extracted features: ', time.time() - loading_time)
    #
    h5file = h5py.File(os.path.join(dir_path, 'my_discrete_arousal_concat.h5'), 'r')
    train_arousal = {}
    for k, v in h5file.items():
        train_arousal[int(k)] = torch.from_numpy(v.value) #.to(device)  # Convert numpy arrays to tensors on gpu  # .to(device)
    h5file.close()

    return train_features, train_arousal

# Main
def main(args):
    # Device configuration
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Manual seed
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device: ', device)

    #------------------------------------------------------------------------------------------------
    # input_size for the 2FC-layer model
    rgb_size = 2048
    OF_size = 2048
    audio_size = 1582
    input_size = rgb_size + OF_size + audio_size
    #-----------------------------------------------------------------------------------------------
    # Cross-validation
    print('Cross-validation.....')
    Accuracy_ave = 0
    Accuracy_1_ave = 0

    movlistlength = len(movlist)

    for index in range(0, movlistlength):
        m_start_time = time.time()

        # Build the model
        model = Two_FC_layer().to(device)

        # Loss and optimizer
        # Cross Entropy Loss
        criterion = nn.CrossEntropyLoss()
        #---------------------------------------------------------------------------------
        # KL Loss
        # criterion = nn.KLDivLoss()
        #---------------------------------------------------------------------------------
        optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.dw)  # 0.05

        # for model training
        train_features, train_arousal = loadingfiles(device)

        # for model validation
        validate_features = train_features[index].clone()
        validate_arousal = train_arousal[index].clone()

        # for model training
        train_features.pop(index)
        train_arousal.pop(index)

        #
        train_dataset = train_dataloader_for_FC_model_Arousal(train_features, train_arousal, args)
        # validate_dataset = validate_dataloader_for_FC_model_Arousal(validate_features, validate_arousal, validate_cont_arousal, args)

        # Train and validate on each epoch
        print('Validate on: ', movlist[index],'. Train on the rest.')

        model, train_losses, valid_losses = train_func(train_dataset, validate_features, validate_arousal, model, device, criterion, optimizer, args.num_epochs, input_size, args.patience)
        print('Training time for ', movlist[index], ': ', time.time() - m_start_time)

        val_output_disc, val_accuracy, val_accuracy_1 = validate_func(validate_features, validate_arousal, model, device)

        Accuracy_ave += val_accuracy
        Accuracy_1_ave += val_accuracy_1
        #----------------------------------------------------------------------------------------------------------
        # Save model
        # Model name
        model_name = movlist[index] + '_emobase2010_2FC_Arousal_audio_video.pth'
        torch.save(model.state_dict(), os.path.join(args.model_path, model_name))

        #---------------------------------------------------------------------------------------------------------------
        # save predicted arousal labels
        afilename = movlist[index] + '_predArousal_emobase2010_2FC_classification_audio_video.h5'
        h5file = h5py.File(os.path.join(pred_path, afilename), mode='w')
        savedata = val_output_disc.cpu()
        h5file.create_dataset('default', data=np.array(savedata.detach().numpy(), dtype=np.int32))
        h5file.close()


        # Free memory
        del model, optimizer, validate_features, validate_arousal, val_output_disc, train_features, train_arousal
        freeCacheMemory()
        #
        print('Running time for ', movlist[index], ' : ', time.time() - m_start_time)
        print('After validation: ')
        memoryCheck()

    Accuracy_1_ave += Accuracy_ave
    print('-----------------------------------------------RESULTS----------------------------------------------- \n')
    print('12-fold cross-validation: ')

    print('For discrete case: Arousal: Accuracy: {:.5f}, Accuracy+/-1: {:.5f} \n'.format(
            100 * Accuracy_ave / movlistlength, 100 * Accuracy_1_ave / movlistlength))



if __name__ == "__main__":
    #
    dir_path = '/home/minhdanh/Documents/2FC_RGB_OF_Audio'
    model_path = os.path.join(dir_path, 'Thao_model')        # path to save models
    pred_path = os.path.join(dir_path, 'PredictedValues')    # path to save predicted arousal values
    # ------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default= model_path, help='path for saving trained models')
    #-------------------------------------------------------------------------------------------------------------------

    parser.add_argument('--num_epochs', type=int, default=200)  # 200
    parser.add_argument('--patience', type=int, default=25, help ='early stopping patience; how long to wait after last time validation loss improved')

    parser.add_argument('--batch_size', type=int, default=128, help = 'number of feature vectors loaded per batch')  #128
    parser.add_argument('--lr', type=float, default = 0.005, metavar='LR', help = 'initial learning rate')
    parser.add_argument('--dw', type=float, default = 0.005, metavar='DW', help = 'decay weight')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 123)')

    args = parser.parse_args()
    print(args)

    # ------------------------------------------------------------------------------------------------------------------
    movlist = ['BMI', 'LOR', 'GLA', 'DEP', 'CRA', 'CHI', 'FNE', 'ABE', 'MDB', 'NCO', 'RAT', 'SIL']

    # Temperature in softmax
    T = 2.0
    # Means of bins:
    num_bins = 7
    step = 2.0 / num_bins
    bin_means = np.array([np.float32(-1.0 + step / 2.0)])
    for i in range(1, num_bins):
        binmean = (-1.0 + step / 2.0) + i * step
        bin_means = np.append(bin_means, np.float32(binmean))
    #-------------------------------------------------------------------------------------------------------------------
    # Note: OF_image_names.csv and image-values.csv must have the same row numbers (number of opt. flow images = numb of images)
    main_start_time = time.time()
    main(args)
    print('Total running time: {:.5f} seconds' .format(time.time() - main_start_time))
