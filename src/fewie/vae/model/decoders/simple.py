import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, dims):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(dims[2], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[0])
        self.float()
    def forward(self, z):
        y= self.fc1(z)
        y= self.fc2(y)
        y=torch.sigmoid(y)
        #postprocessing
        return y.float()#.reshape((-1, 1, 28,28)) # reshape if flattened for linear...
