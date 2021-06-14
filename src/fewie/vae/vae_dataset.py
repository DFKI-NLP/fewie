import torch
import numpy as np
from torch.utils.data import Dataset
class VaeDataset(Dataset):
    def __init__(self, dataset, target_name,sentence_length, transform=None):#, target_transform=None):
        self.dataset=dataset
        self.target_name=target_name
        self.sentence_length=sentence_length
        self.transform=transform
       # self.target_transform=target_transform
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pos_matrix=np.array([example[self.target_name] for example in self.dataset])
        max_length=np.max([len(row)for row in pos_matrix])
        data= np.squeeze(np.array([[np.pad(row, (0, max_length-len(row)), 'constant', constant_values=0)]for row in pos_matrix]), axis=1)[:,:self.sentence_length]
        inp=data[idx]
      #  outp=data[idx]
        if self.transform:
            inp=self.transform(inp)
#        if self.target_transform:
 #           outp=self.target_transform(outp)

        return inp#, outp
