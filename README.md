# Predicting emotion of viewers evoked by movie clips. 

The goal of this project is to  develop  and  analyze  multimodal  models  for  predicting  experienced  affective  responses of viewers watching movie clips. We develop hybrid multimodal prediction models based on both the video and audio of the clips. For the video content, we extract features from RGB frames and optical flow using pretrained neural networks.  For the audio model, we compute  an  enhanced  set  of  low-level  descriptors  including  intensity,  loudness,  cepstrum,  linear  predictor  coefficients, pitch and voice quality.  Both visual and audio features are then concatenated to create audio-visual features, which are used to predict the evoked emotion. To classify the movie clips into the corresponding affective response categories, we propose two approaches based on deep neural network models.  The first one is based on fully connected layers without memory on the time component, the second incorporates the sequential dependency with an LSTM. 

## Prerequisites

The code snippets were implemented in Ubuntu 18, Python 3.6 and the experiments were run on a NVIDIA GTX 1070. 

## Data

We use the [extended Cognimuse dataset](http://cognimuse.cs.ntua.gr/database)

* Spatial input data: 
We extract RGB frames from each movie clips in the dataset and save as .jpg images in disk. 
We use ResNet-50 model, except the last fully connected layer, pre-trained on ImageNet to extract spatial features from RGB frames. 

* Motion input data: 
We use [PWC-Net](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch) to extract optical flow fields from RGB frames and save as .jpg images in disk. 
We use a stack of 10 sequential optical flows as the input to the ResNet-101 model that has been pre-trained on the ImageNet classification task. The first convolutional layer and the last classification layer, which had been fine-tuned to be able to ingest 10 stacks of sequential optical flows to predict action recognition on UCF-101. Pre-trained model is available at [here](https://github.com/jeffreyhuang1/twostream-action-recognition). 
We remove the last fully connected classification layer in the ResNet-101 model and freeze the rest.

* Audio input data:
Audio features are extracted using the OpenSMILE toolkit with a frame size of 400ms with a hop size of 40ms. The frame size corresponds to a
time period of a stack of 10 optical flows. We use a configuration file named “emobase2010”, which is based on INTERSPEECH 2010 paralinguistics
challenge. 

## Training and testing

We do leave-one-out cross-validation on the extended Cognimuse dataset including 12 movie clips, therefore, we have 12 pre-trained models. The uploaded pre-trained models saved in *.path* files are the best ones.
 
## Paper & Citation

If you use this code, please cite the following paper: 

@article{,
  author={},
  title={Multimodal Deep Models for Predicting Affective Responses Evoked by Movies},
  journal={Proceedings of the 2nd International Workshop on Computer Vision for Physiological Measurement, as part of ICCV. Seoul, South Korea},
  year={2019}
}

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


