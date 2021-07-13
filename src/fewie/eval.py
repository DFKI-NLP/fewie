from typing import Any, Dict
import csv

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
import datasets
from datasets import load_dataset
from fewie.data.datasets.utils import get_label_to_id
from fewie.evaluation.scenarios.few_shot_linear_readout import eval_few_shot_linear_readout
from fewie.evaluation.utils import seed_everything
from fewie.vae.train_vae import pretrain_vae
from fewie.vae.vae_dataset import VaeDataset
import torchvision
from torchvision.transforms import transforms
def evaluate_config(cfg: DictConfig) -> Dict[str, Any]:
    seed_everything(cfg.seed)

    device = torch.device("cuda", cfg.cuda_device) if cfg.cuda_device > -1 else torch.device("cpu")

    dataset = instantiate(cfg.dataset)
    #print(dataset.shape)
    #print(len(dataset))
    #return
    #print(dataset['train'].features)
    #print(type(dataset['train'].features))
    #print(dataset['train'][0])
    #return
    #print(dataset)
    #print(dataset['pos_tags'][0])
   # return
    if isinstance(dataset, datasets.DatasetDict):
        label_to_id = get_label_to_id(dataset["train"].features, cfg.label_column_name)
    else:
        label_to_id = get_label_to_id(dataset.features, cfg.label_column_name)
    #pos_matrix=np.array([example['pos_tags'] for example in dataset] )
    #print('pos matrix shape:',pos_matrix.shape)
    #max_length=np.max([len(row) for row in pos_matrix])
    #print('length:', max_length)
    #max_padded_matrix=np.squeeze(np.array([[np.pad(row,(0,max_length-len(row)),'constant',constant_values=0)] for row in pos_matrix]), axis=1)
    #print('max padded shape:',max_padded_matrix.shape)
    #limited_matrix=max_padded_matrix[:, :30]#limit of 30 words is set arbitrarily
    #print('final matrix shape:',limited_matrix.shape)
    #transformations=transforms.Compose([
    #    transforms.ToTensor()
    #    ])
    
    #print(dataset['pos_tags'][0])
    #print(type(dataset['pos_tags']))
    #print(type(dataset['pos_tags'][0]))
    if cfg.use_vae:

        print('using vae support')
        transformations=[]
        print('preparing dataset...')
        vae_dataset=VaeDataset(dataset, 'pos_tags', 30, transformations)#, transformations])
        print('starting training...')
        model=pretrain_vae(vae_dataset, dims=[30,20,10], epochs=cfg.vae_epochs,batch_size=128, model_name='test',device=device)#training not randomized yet, TODO
        print('done, predicting tags...')
        extracted_data=extract_data(dataset, 'pos_tags',30)
        predictions=np.argmax(model(torch.from_numpy(np.array(extracted_data)).to(device)).cpu().detach().numpy(),axis=1)
        print('exporting tags...')
        
         
        data=[]
        for i in range(10):#len(dataset)):# REDUCED FOR TESTING
             for j, label in enumerate(dataset.features):
                 if j==0:
                       data.append({label:dataset[label][i]})
                 elif j==len(dataset.features)-1:
                       data[-1][label]=dataset[label][i]
                       data[-1]['vae_tag']=[predictions[i]]*len(dataset[label][i])
                       
                 else:
                       data[-1][label]=dataset[label][i]

    #for i,label in enumerate(dataset.features):
        #for j, content in enumerate(dataset[label])
        #data.append(dataset[label])
     #   data=dataset[label]
    #print(data[list(data.keys())[1]])
    #print(list(data.keys()))
    #fieldn=dataset.features
    #print(np.array(data).shape)
        _fieldnames=list(dataset.features)
        _fieldnames.append('vae_tag')
        with open('/netscratch/mikkelsen/code/fewie/src/fewie/data/datasets/'+cfg.dataset.path+'_'+cfg.scenario.name+'.csv','w') as csvfile:#no safetymeasure here, path hardcoded! TODO
             dictwriter=csv.DictWriter(csvfile, fieldnames=_fieldnames)#dataset.features)
             dictwriter.writeheader()
             dictwriter.writerows(data)

        print('done. returning to evaluation...')
    
    
    
    #return  
    #dataset['pos_tags']=np.zeros(len(dataset['pos_tags']))
    #print(dataset.shape)
    dataset_processor = instantiate(
        cfg.dataset_processor,
        label_to_id=label_to_id,
        text_column_name=cfg.text_column_name,
        label_column_name=cfg.label_column_name,
    )
    # assuming intendet vae usage
    print(dataset.shape)
    #model=pretrain_vae(dataset, dims=( 


    encoder = instantiate(cfg.encoder)
    encoder = encoder.to(device)

    classifier = instantiate(cfg.evaluation.classifier)

 #   processed_dataset = dataset_processor(dataset)
    processed_dataset=load_dataset('csv', data_files='/netscratch/mikkelsen/code/fewie/src/fewie/data/datasets/'+cfg.dataset.path+'_'+cfg.scenario.name+'.csv')
#    print(processed_dataset['pos_tags'][0])

#    print(aset.shape)
#    return
#    return
    if cfg.scenario.name == "few_shot_linear_readout":
        few_shot_dataset = instantiate(
            cfg.evaluation.dataset,
            dataset=processed_dataset,
            columns=dataset_processor.feature_columns,
            label_column_name=cfg.label_column_name,
        )

        evaluation_results = eval_few_shot_linear_readout(
            classifier=classifier,
            encoder=encoder,
            dataset=processed_dataset,
            few_shot_dataset=few_shot_dataset,
            device=device,
            batch_size=cfg.batch_size,
            metrics=cfg.scenario.metrics,
        )
    else:
        raise ValueError("Unknown evaluation scenario '%s'" % cfg.scenario)

    return evaluation_results

def extract_data(data, tag, sentence_length):
    # extracts data on the fly
    raw=[x for x in data[tag]]
    max_length=np.max([len(row)for row in raw])
    _data=np.squeeze(np.array([[np.pad(row, (0, max_length-len(row)), 'constant', constant_values=0)] for row in raw]), axis=1)[:, :sentence_length]
    return _data
