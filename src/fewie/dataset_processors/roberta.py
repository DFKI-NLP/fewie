import random
from typing import Dict, List, Union

from transformers import AutoTokenizer

import datasets
from fewie.dataset_processors.processor import DatasetProcessor


class RobertaProcessor(DatasetProcessor):
    def __init__(
        self,
        tokenizer_name_or_path: str,
        text_column_name: str,
        label_column_name: str,
        label_to_id: Dict[str, int],
        max_length: int = 128,
        label_all_tokens: bool = False,
        padding: str = "max_length",
        add_prefix_space: bool = True,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, add_prefix_space=True
        )
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.label_to_id = label_to_id
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens
        self.padding = padding
        self.prefix_space = add_prefix_space

    @property
    def feature_columns(self) -> List[str]:
        return ["input_ids", "attention_mask", "labels"]

    def __call__(
        self, dataset: Union[datasets.Dataset, datasets.DatasetDict]
    ) -> Union[datasets.Dataset, datasets.DatasetDict]:
        return dataset.map(self.tokenize_and_align_labels, batched=True)

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples[self.text_column_name],
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            # roberta requirement:
            #            add_prefix_space=True
        )

        labels = []
        for i, label in enumerate(examples[self.label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(
                        self.label_to_id[label[word_idx]]
                        if self.label_all_tokens
                        else -100
                    )
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
