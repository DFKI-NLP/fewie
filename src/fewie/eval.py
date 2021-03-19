from typing import Any, Dict

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from fewie.evaluation.scenarios.few_shot_linear_readout import eval_few_shot_linear_readout
from fewie.evaluation.utils import seed_everything


def evaluate_config(cfg: DictConfig) -> Dict[str, Any]:
    seed_everything(cfg.seed)

    device = torch.device("cuda", cfg.cuda_device) if cfg.cuda_device > -1 else torch.device("cpu")

    dataset = instantiate(cfg.dataset)

    dataset_processor = instantiate(cfg.dataset_processor)

    encoder = instantiate(cfg.encoder)
    encoder = encoder.to(device)

    classifier = instantiate(cfg.evaluation.classifier)

    processed_dataset = dataset_processor(dataset)

    if cfg.scenario.name == "few_shot_linear_readout":
        few_shot_dataset = instantiate(
            cfg.evaluation.dataset,
            dataset=processed_dataset,
            columns=dataset_processor.feature_columns,
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
