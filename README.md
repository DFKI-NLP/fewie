# FewIE: Few-shot named entity recognition

## 🚀&nbsp; Installation

### Prerequisites

Install [PyTorch](https://pytorch.org/get-started).

### From source
```bash
git clone https://github.com/ChristophAlt/fewie
cd fewie
pip install .
```

### For development
```bash
git clone https://github.com/ChristophAlt/fewie
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

```

For example to run the evaluation on  with a baseline BERT encoder, run the following command:
```python

```

This should produce an output similar to this:
```

```

# TODOs
- [ ] Clarify the best approach to produce prototypes given token embeddings. Input -> Encoder -> embeddings [N x K x d_hidden] -> Pooling (select and aggregate token embeddings for each N) -> Prototypes [N x d_hidden]
    - Solution 1: Encoder with a forward and pooling step that produces the prototypes (if mask, or labels plus label id, are provided) and returns prototypes [N x d_hidden] (optional) and final hidden states [N x K x d_hidden].
- [ ] Clarify the evaluation setting, as the survey paper is missing a lot of important details.
