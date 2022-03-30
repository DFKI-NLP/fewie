import datasets
import json


_CITATION = """
@inproceedings{zhang-etal-2020-bootstrapping,
    title = "Bootstrapping Named Entity Recognition in {E}-Commerce with Positive Unlabeled Learning",
    author = "Zhang, Hanchu  and
      Hennig, Leonhard  and
      Alt, Christoph  and
      Hu, Changjian  and
      Meng, Yao  and
      Wang, Chao",
    booktitle = "Proceedings of The 3rd Workshop on e-Commerce and NLP",
    month = jul,
    year = "2020",
    address = "Seattle, WA, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.ecnlp-1.1",
    doi = "10.18653/v1/2020.ecnlp-1.1",
    pages = "1--6",
    abstract = "In this work, we introduce a bootstrapped, iterative NER model that integrates a PU
    learning algorithm for recognizing named entities in a low-resource setting. Our approach combines
    dictionary-based labeling with syntactically-informed label expansion to efficiently enrich the
    seed dictionaries. Experimental results on a dataset of manually annotated e-commerce product
    descriptions demonstrate the effectiveness of the proposed framework.",
}
"""

_DESCRIPTION = """
Lenovo NER dataset.
"""

# count: 20453, 1485, 443, 3234, 1493
NER_TAGS_DICT = {
    "O": 0,
    "ATTRIBUTE": 1,
    "BRAND": 2,
    "COMPONENT": 3,
    "PRODUCT": 4,
}

# count: 20453, 717, 768, 423, 20, 1246, 1988, 428, 1065
NER_BIO_TAGS_DICT = {
    "O": 0,
    "B-ATTRIBUTE": 1,
    "I-ATTRIBUTE": 2,
    "B-BRAND": 3,
    "I-BRAND": 4,
    "B-COMPONENT": 5,
    "I-COMPONENT": 6,
    "B-PRODUCT": 7,
    "I-PRODUCT": 8,
}


class LenovoConfig(datasets.BuilderConfig):
    """BuilderConfig for Lenovo"""

    def __init__(self, **kwargs):
        """BuilderConfig for Lenovo dataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(LenovoConfig, self).__init__(**kwargs)


class Lenovo(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.features.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.features.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "ATTRIBUTE", "BRAND", "COMPONENT", "PRODUCT"]
                        )
                    ),
                    "ner_bio_tags": datasets.features.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-ATTRIBUTE",
                                "I-ATTRIBUTE",
                                "B-BRAND",
                                "I-BRAND",
                                "B-COMPONENT",
                                "I-COMPONENT",
                                "B-PRODUCT",
                                "I-PRODUCT",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": self.config.data_files},
            ),
        ]

    def _generate_examples(self, filepath=None):
        """Yields examples."""
        filepath = filepath["train"][0]
        with open(filepath, encoding="utf-8") as f:
            for line in f.readlines():
                record = json.loads(line)
                id_ = record["id"]
                yield id_, record
