import torch
from torch import nn
from torch.nn import functional as F
import torch



class Two_FC_layer(torch.nn.Module):
    def __init__(self, input_dim = 2048, reduced_dim=128, fc_dim = 64, num_classes=7):
        super(Two_FC_layer, self).__init__()
        self.reduced_rgb = nn.Linear(input_dim, reduced_dim, bias=False)

        self.fc1 = nn.Linear(reduced_dim, fc_dim, bias=False)
        self.fc2 = nn.Linear(fc_dim, fc_dim, bias=False)
        self.class_dim = nn.Linear(fc_dim, out_features=num_classes, bias=False)  # output

    def forward(self, x):
        out = self.class_dim(self.fc2(self.fc1(self.reduced_rgb(x))))
        return out




