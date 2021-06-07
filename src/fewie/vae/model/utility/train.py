import torch
import torch.nn as nn
import numpy as np
def train(model, data,device, epochs=1):
    mse=nn.MSELoss()
    adam=torch.optim.Adam(model.parameters())
    sampling_interval=int(epochs/10) if int(epochs/10)>0 else 1#add manual mode
    for epoch in range(epochs):
        losses=[]
        for _x,_ in data:
            adam.zero_grad()
            x=_x.to(device)
            #print(list(x.size()))
            prediction=model(x)
            loss=mse(prediction, x)+model.encoder.kl_div
            losses.append(loss.item())
            loss.backward()
            adam.step()
        if epoch%sampling_interval==0:
            print('Training Epoch: {}/{}, Loss is: {.0f}%)'.format( epoch, len(data), np.mean(losses)))
    return model
