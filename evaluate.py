import json
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from fewie.eval import evaluate_config

logger = logging.getLogger(__name__)


@hydra.main(config_name="config", config_path="config")
def evaluate(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    evaluation_results = evaluate_config(cfg)
    logger.info("Evaluation results:\n%s" % evaluation_results)
    with open("./evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f)


if __name__ == "__main__":
    evaluate()
