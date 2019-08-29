import torch
from torch import nn
from torch.nn import functional as F
import torch



class Two_FC_layer(torch.nn.Module):
    def __init__(self, rgb_dim=2048, OF_dim = 2048, audio_dim = 1582, reduced_dim=128, fc_dim = 64, num_classes=7):
        super(Two_FC_layer, self).__init__()
        self.reduced_rgb = nn.Linear(rgb_dim, reduced_dim, bias=False)
        self.reduced_OF = nn.Linear(OF_dim, reduced_dim, bias=False)
        self.reduced_audio = nn.Linear(audio_dim, reduced_dim, bias=False)
        self.rgb = rgb_dim
        self.OF = OF_dim
        self.audio = audio_dim

        self.fc1 = nn.Linear(3*reduced_dim, fc_dim, bias=False)
        self.fc2 = nn.Linear(fc_dim, fc_dim, bias=False)
        self.class_dim = nn.Linear(fc_dim, out_features=num_classes, bias=False)  # output

    def forward(self, x):
        temp = torch.cat((self.reduced_rgb(x[:, 0:self.rgb]), self.reduced_OF(x[:, self.rgb : (self.rgb+self.OF)]),
                          self.reduced_audio(x[:, (self.rgb+self.OF):(self.rgb+self.OF+self.audio)])), dim=1)
        out = self.class_dim(self.fc2(self.fc1(temp)))
        return out




