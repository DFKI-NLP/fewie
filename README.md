# FewIE: Few-shot named entity recognition

## 🚀&nbsp; Installation

### Prerequisites

Install [PyTorch](https://pytorch.org/get-started).

### From source
```bash
git clone https://github.com/DFKI-NLP/fewie
cd fewie
pip install .
```

### For development
```bash
git clone https://github.com/DFKI-NLP/fewie
cd fewie
pip install -e .
```

## 💡&nbsp; Usage

To run the default experiment setting, run:
```python
python evaluate.py
```

To show the available options and the default config, do:
```python
python evaluate.py --help
```

which results in something like this:
```
== Configuration groups ==
Compose your configuration from those groups (group=option)

dataset: conll2003
dataset_processor: bert, spanbert, transformer
encoder: bert, random, spanbert, transformer
evaluation/classifier: logistic_regression
evaluation/dataset: nway_kshot, nway_kshot_5_1, nway_kshot_na_dedicated, nway_kshot_na_rest


== Config ==
Override anything in the config (foo.bar=value)

dataset:
  _target_: datasets.load_dataset
  path: conll2003
  version: 1.0.0
  split: test
dataset_processor:
  _target_: fewie.dataset_processors.transformer.TransformerProcessor
  tokenizer_name_or_path: ??
  max_length: 128
  label_all_tokens: false
encoder:
  _target_: fewie.encoders.random.RandomEncoder
  embedding_dim: 768
evaluation:
  dataset:
    _target_: fewie.data.datasets.generic.nway_kshot.NwayKshotDataset
    n_ways: 5
    k_shots: 1
    n_queries: 1
    n_samples: 600
    deterministic: false
  classifier:
    _target_: fewie.evaluation.classifiers.logistic_regression.LogisticRegression
    C: 1.0
    penalty: l2
    random_state: 0
    solver: lbfgs
    max_iter: 1000
    multi_class: multinomial
seed: 1234
cuda_device: 0
batch_size: 1
text_column_name: tokens
label_column_name: ner_tags
scenario:
  name: few_shot_linear_readout
  metrics:
  - accuracy
  - f1_micro
  - f1_macro
```

For example to run the evaluation on CoNLL 2003 with a baseline BERT encoder, run the following command:
```sh
python evaluate.py \
    dataset=conll2003 \
    dataset_processor=bert \
    encoder=bert \
    evaluation/dataset=nway_kshot_5_1
```

This should produce an output similar to this:
```json
{'accuracy': {
    'mean': 0.5341624247825408, 'margin_of_error': 0.019547702016139496, 'confidence': 0.95
    }, 
    'f1_micro': {'mean': 0.5341624247825408, 'margin_of_error': 0.019547702016139496, 'confidence': 0.95
    }, 
    'f1_macro': {'mean': 0.23580803312291498, 'margin_of_error': 0.010821995530256272, 'confidence': 0.95}}
```
