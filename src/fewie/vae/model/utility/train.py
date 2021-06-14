import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
def train(model, data,device,batch_size, epochs=1):
    mse=nn.MSELoss()
    dataloader=DataLoader(x, batch_size=batch_size, shuffle=True)
    adam=torch.optim.Adam(model.parameters())
    sampling_interval=int(epochs/10) if int(epochs/10)>0 else 1#add manual mode
    for epoch in range(epochs):
        losses=[]
        for _x in dataloader:   
            adam.zero_grad()
            #_x=_x.astype('double')
            #x=torch.from_numpy(_x).to(device)
            x=_x[0].to(device)
            #print(list(x.size()))
            prediction=model(x)
            loss=mse(prediction, x)+model.encoder.kl_div
            losses.append(loss.item())
            loss.backward()
            adam.step()
        if epoch%sampling_interval==0:
            print('Training Epoch: {}/{}, Loss is: {.0f}%)'.format( epoch, len(data), np.mean(losses)))
    return model
