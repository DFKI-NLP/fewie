
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
#torch.use_deterministic_algorithms(True) # forcing greatest possible level of reproducibility
torch.backends.cudnn.deterministic = True


import torch.nn as nn
import torch.nn.functional
import torch.utils
import torch.distributions
#import torchvision

#from model.encoders import *
#from model.decoders import *
from fewie.vae.model.utility.train import train
from fewie.vae.model.vae import VAE


#print('imports working')
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device='cpu'
def pretrain_vae(data, dims, epochs: int,model_name: str, reuse: bool=False):
    device='cude' if torch.cuda.is_available() else 'cpu'
    _model=VAE(dims)
    model=_model.to(device)
    if os.path.exists('model/pretrained/'+model_name+'.pth'):
        print('model already exists, using pretrained...')
        model.load_state_dict(torch.load('model/pretrained/'+model_name+'.pth'))
        return model.eval()

    else:
        model=train(model, data,device,epochs)
        print('model pretrained, creating backup...')
        torch.save(model.state_dict(), 'model/pretrained/'+model_name+'.pth')
        return model.eval()


# hidden_dim_size= [784,512,10] # adapt to arch!
# epochs=10
#
# _model=VAE(hidden_dim_size)
# model=_model.to(device)
# print('model initialized')
# data=torch.utils.data.DataLoader(
#         torchvision.datasets.MNIST('./data',
#                transform=torchvision.transforms.ToTensor(),
#                download=True),
#         batch_size=128,
#         shuffle=True)
# print('data initialized')
# model=train(model, data,device,epochs)
# print('training worked')
# torch.save(model.state_dict(), 'trained_models/simple_mnist.pth')
# print('exported model')
