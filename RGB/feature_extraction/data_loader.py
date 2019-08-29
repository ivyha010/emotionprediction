import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
import os

# Dataset
class CognimuseDataset(data.Dataset):
    """COGNIMUSE Custom Dataset """
    def __init__(self, csv_path_file, root_dir,transform=None):
        """Set the path for images,...
        Args:
            image_path_file: path to the txt file with valence /arousal values
            root_dir: directory with all images.
            transform (callable, optional): image transformer. Optional transform to be applied on a sample.
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path_file, header = None)
        # First column contains the image names
        self.image_arr = np.asarray(self.data_info.iloc[:,0])
        # Second column is valence values
        self.valence_arr = np.asarray(self.data_info.iloc[:,1]) # valence
        # Third column is arousal values
        self.arousal_arr = np.asarray(self.data_info.iloc[:,2]) # arousal
        #
        self.root_dir = root_dir
        #
        self.transform_resize = transforms.Resize(size=500, interpolation=Image.NEAREST) # output is size*H/W, size
        self.transform = transform
        # Calculate len
        self.data_len = len(self.data_info.index)

    # __getitem__: dataset[i] can return the ith datapoint
    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        # Get image name from the pandas df
        single_img_name = os.path.join(self.root_dir, self.image_arr[index])
        # Open image
        image_as_image = Image.open(single_img_name).convert('RGB')
        # resize
        image_as_image = self.transform_resize(image_as_image)
        # Get valence value/label(class) of the image based on the cropped pandas column
        single_valence_value = self.valence_arr[index]
        # Get arousal value/label(class) of the image based on the cropped pandas column
        single_arousal_value = self.arousal_arr[index]
        # Transform image to tensor
        if self.transform is not None:
            image_as_tensor = self.transform(image_as_image)
        else: image_as_tensor= self.to_tensor(image_as_image)
        return (image_as_tensor, single_valence_value, single_arousal_value)

    # len(dataset) returns the size of the dataset
    def __len__(self):
        return self.data_len  # returns how many images we have

