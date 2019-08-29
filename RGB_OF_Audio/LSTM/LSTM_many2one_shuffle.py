import torch
from torch import optim, nn
import torch.nn.functional as F

class many2one_LSTM(torch.nn.Module):
    def __init__(self, rgb_dim=2048, OF_dim = 2048, audio_dim = 1582, reduced_dim=128, hidden_dim = 64, num_layers = 2, num_classes=7):  
        super(many2one_LSTM, self).__init__()
        self.reduced_rgb = nn.Linear(rgb_dim, reduced_dim, bias=False)
        self.reduced_OF = nn.Linear(OF_dim, reduced_dim, bias=False)
        self.reduced_audio = nn.Linear(audio_dim, reduced_dim, bias=False)
        self.rgb = rgb_dim
        self.OF = OF_dim
        self.audio = audio_dim

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(3*reduced_dim, hidden_dim, num_layers, batch_first= True)
        self.class_dim = nn.Linear(hidden_dim, num_classes)  #, bias=False)  # 128, 64

    def forward(self, x):  # x: featureseqs
        temp = torch.cat((self.reduced_rgb(x[:, :, 0:self.rgb]), self.reduced_OF(x[:, :, self.rgb: (self.rgb + self.OF)]),
                          self.reduced_audio(x[:, :, (self.rgb + self.OF):(self.rgb + self.OF + self.audio)])), dim=2)

        # Set initial hidden and cell states
        h0 = torch.zeros([self.num_layers, temp.shape[0], self.hidden_dim])  # , requires_grad=False)
        c0 = torch.zeros([self.num_layers, temp.shape[0], self.hidden_dim])  # , requires_grad=False)
        #
        h0, c0 = h0.cuda(), c0.cuda()
        #
        # Forward propagate LSTM
        out, _ = self.lstm.forward(temp, (h0, c0))  # out: tensor of shape (batch, seq_length, hidden_size)
        #out, _ = self.lstm.forward(x, (h0, c0))

        # Outputs: many2one
        out = self.class_dim(out[:, -1, :])   # choose the last one
        # out = self.class_dim(out.mean(dim=1))  # averaging
        return out

