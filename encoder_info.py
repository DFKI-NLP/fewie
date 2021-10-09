import torch
from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm
import csv


MODEL_NAMES = [
    # for English
    "bert-base-uncased",
    "bert-base-cased",
    "albert-base-v2",
    "roberta-base",
    "SpanBERT/spanbert-base-cased",
    "xlnet-base-cased",
    # for German
    "dbmdz/bert-base-german-uncased",
    "uklfr/gottbert-base",
    "xlm-roberta-base",
]


if __name__ == "__main__":
    records = []

    for model_name in tqdm(MODEL_NAMES):
        # get vocab size
        if model_name in ["roberta-base", "uklfr/gottbert-base", "xlm-roberta-base"]:
            tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        vocab_size = tokenizer.vocab_size

        # get model size
        model = AutoModel.from_pretrained(model_name)
        model_size = model.num_parameters()

        record = {
            "model_name": model_name,
            "vocab_size": vocab_size,
            "model_size": model_size,
        }
        print(record)
        records.append(record)

    # write into .csv file
    column_names = records[0].keys()
    with open("encoder-info.csv", "w", newline="") as fw:
        dict_writer = csv.DictWriter(fw, column_names)
        dict_writer.writeheader()
        dict_writer.writerows(records)
