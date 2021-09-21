import datasets
from tqdm import tqdm
import os


_DESCRIPTION = """
CoNLL German dataset. 
12152 texts in train, 2867 in test(testa) and 3005 in val(testb).
"""

# the label ids
NER_TAGS_DICT = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
    "B-MISC": 7,
    "I-MISC": 8,
}


class CoNLLConfig(datasets.BuilderConfig):
    """BuilderConfig for CoNLL(de)."""

    def __init__(self, **kwargs):
        """BuilderConfig for CoNLL(de).
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CoNLLConfig, self).__init__(**kwargs)


class CoNLL(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.features.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.features.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-PER",
                                "I-PER",
                                "B-LOC",
                                "I-LOC",
                                "B-ORG",
                                "I-ORG",
                                "B-MISC",
                                "I-MISC",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(self.config.data_files, "deuutf.train")
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(self.config.data_files, "deuutf.testa")
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(self.config.data_files, "deuutf.testb")
                },
            ),
        ]

    def _generate_examples(self, filepath=None):
        num_lines = sum(1 for _ in open(filepath))
        id = 0

        with open(filepath, "r") as f:
            tokens, ner_tags = [], []
            for line in tqdm(f, total=num_lines):
                line = line.strip().split()

                if line:
                    assert len(line) == 5
                    token, _, _, _, ner_tag = line

                    if token == "-DOCSTART-":
                        continue

                    tokens.append(token)
                    ner_tags.append(NER_TAGS_DICT[ner_tag])

                elif tokens:
                    # organize a record to be written into json
                    record = {
                        "id": str(id),
                        "tokens": tokens,
                        "ner_tags": ner_tags,
                    }
                    tokens, ner_tags = [], []
                    id += 1
                    yield record["id"], record

            # take the last sentence
            if tokens:
                record = {
                    "id": str(id),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }
                yield record["id"], record
