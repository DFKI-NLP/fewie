from typing import Any, Dict

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
import datasets
from fewie.data.datasets.utils import get_label_to_id
from fewie.evaluation.scenarios.few_shot_linear_readout import eval_few_shot_linear_readout
from fewie.evaluation.utils import seed_everything
from fewie.vae.train_vae import pretrain_vae

def evaluate_config(cfg: DictConfig) -> Dict[str, Any]:
    seed_everything(cfg.seed)

    device = torch.device("cuda", cfg.cuda_device) if cfg.cuda_device > -1 else torch.device("cpu")

    dataset = instantiate(cfg.dataset)

    if isinstance(dataset, datasets.DatasetDict):
        label_to_id = get_label_to_id(dataset["train"].features, cfg.label_column_name)
    else:
        label_to_id = get_label_to_id(dataset.features, cfg.label_column_name)
    pos_matrix=np.array([example['pos_tags'] for example in dataset] )
    print('pos matrix shape:',pos_matrix.shape)
    max_length=np.max([len(row) for row in pos_matrix])
    print('length:', max_length)
    max_padded_matrix=np.squeeze(np.array([[np.pad(row,(0,max_length-len(row)),'constant',constant_values=0)] for row in pos_matrix]), axis=1)
    print('max padded shape:',max_padded_matrix.shape)
    limited_matrix=max_padded_matrix[:, :20]#limit of 20 words is set arbitrarily
    print('final matrix shape:',limited_matrix.shape)
    model=pretrain_vae(limited_matrix, dims=[20,20,10], epochs=1, model_name='test')#50 % compression?
    return  
   

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

    processed_dataset = dataset_processor(dataset)

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
