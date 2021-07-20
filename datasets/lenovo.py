import datasets
import json

_CITATION = """

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
        filepath = "./lenovo.json"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": filepath},
            ),
        ]

    def _generate_examples(self, filepath=None):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            for line in f.readlines():
                record = json.loads(line)
                id_ = record["id"]
                yield id_, record
