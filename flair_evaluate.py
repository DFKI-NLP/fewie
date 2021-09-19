import json
import logging
import os
from tqdm import tqdm
import numpy as np

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from flair.data import Sentence
from flair.models import SequenceTagger
import sklearn

from fewie.utils import resolve_relative_path


logger = logging.getLogger(__name__)

# dictionary of CoNLL tags that is used to map (flair) tags
# e.g. `LOC`, `PER` to the corresponding 1-8 class-id
_CONLL_TAGS = {
    "PER": 1,
    "ORG": 3,
    "LOC": 5,
    "MISC": 7,
}


@hydra.main(config_name="flair_config", config_path="config")
def evaluate(cfg: DictConfig) -> None:
    """Conducts evaluation given the configuration.

    Given a dataset and a flair NER-tagger, return the token-level F1-score.

    Args:
        cfg: Hydra-format configuration given in a dict.
    """
    resolve_relative_path(cfg=cfg, start_path=os.path.abspath(__file__))
    print(OmegaConf.to_yaml(cfg))

    dataset = instantiate(cfg.dataset)
    tagger = SequenceTagger.load(cfg.tagger)

    labels, predictions, processed = [], [], 0
    for record in tqdm(dataset):
        tokens, ner_tags = record[cfg.text_column_name], record[cfg.label_column_name]
        sentence = Sentence(" ".join(tokens))

        # ignore sentences with special characters that lead flair tokenizer produce
        # extra "tokens", such as: ["Mike", "'s"] (2 tokens) -> "Mike ' s" (3 tokens)
        if len(sentence) != len(tokens):
            continue

        labels += ner_tags
        pred = np.array([0] * len(ner_tags))
        tagger.predict(sentence)
        for span in sentence.get_spans("ner"):
            span_pos = span.position_string.split("-")
            span_start = int(span_pos[0]) - 1
            pred[span_start] = _CONLL_TAGS[span.tag]

            # if an entity composes more than 1 tokens
            if len(span) == 2:
                span_end = int(span_pos[1]) - 1
                pred[(span_start + 1) : (span_end + 1)] = _CONLL_TAGS[span.tag] + 1

        pred = list(pred)
        predictions += pred
        processed += 1

    logger.info(
        "{}/{} of the test sentences are taken for evalutation.".format(
            processed, len(dataset)
        )
    )

    evaluation_results = sklearn.metrics.f1_score(
        labels, predictions, labels=list(range(1, 9)), average="micro"
    )
    logger.info("Evaluation results:\n%s" % evaluation_results)

    with open("./evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f)


if __name__ == "__main__":
    evaluate()
