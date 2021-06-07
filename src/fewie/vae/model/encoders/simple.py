import torch
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self, dims):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc_mu = nn.Linear(dims[1], dims[2])
        self.fc_sig = nn.Linear(dims[1], dims[2])
        self.ReLU=nn.ReLU()
        # this trick is taken from https://avandekleut.github.io/vae/
        self.normal = torch.distributions.Normal(0, 1)
        self.normal.loc = self.normal.loc.cuda()
        self.normal.scale = self.normal.scale.cuda()


        self.kl_div = 0

    def forward(self, x):
        # preprocessing
        x=torch.flatten(x, start_dim=1)
        # model pass
        #print('pre relu:',list(x.size()))
        x= self.ReLU(self.fc1(x))
        #print('post relu:',list(x.size()))
        mu=self.fc_mu(x)
        sigma=torch.exp(self.fc_sig(x))
        # sampling
        z=self.normal.sample(mu.shape)*sigma+mu
        #kl divergence update:
        self.kl_div=(sigma**2+mu**2-(torch.log(sigma)+.5)).sum()
        return z
