from typing import Dict, Optional, Union, List

import datasets
from fewie.data.datasets import NER_DATASETS_ROOT


def load_dataset(
    dataset_name: str,
    split: str,
    version: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Union[datasets.Dataset, datasets.DatasetDict]:
    """Loads a dataset with a user-customized loading-script.

    Args:
        dataset_name: The name of the loading script, without extension name.
        split: The name of the split, usually be "train", "test" or "validation".
        version: The version of the dataset.
        data_dir: The directory where the dataset is stored.

    Returns:
        The split of the given dataset, in HuggingFace format.
    """
    dataset_script = (NER_DATASETS_ROOT / dataset_name).with_suffix(".py")
    return datasets.load_dataset(
        str(dataset_script), version, data_dir=data_dir, split=split
    )


def get_label_list(labels: List[List[int]]) -> List[int]:
    """Gets a sorted list of all the unique labels from `labels`.

    Args:
        labels: A list of lists, each corresponding to the label-sequence of a text.

    Returns:
        All the unique labels the ever appear in `labels`, given in a sorted list.

    Example:
        Given `labels=[[0, 0, 3, 2, 5], [4, 0], [5, 2, 3]]`, returns `[0, 2, 3, 4, 5]`.
    """
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def get_label_to_id(
    dataset: datasets.Dataset, label_column_name: str
) -> Union[Dict[str, int], Dict[int, int]]:
    """Returns a dictionary the encodes labels to ids, namely integers starting from 0.

    Args:
        dataset: A HuggingFace-format dataset with column `label_column_name`.
        label_column_name: The name of the column, in this case `ner_tags` usually.

    Returns:
        A label-to-id dictionary that maps labels to {0, 1, ..., #classes}.
    """
    if isinstance(dataset.features[label_column_name].feature, datasets.ClassLabel):
        label_list = dataset.features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(dataset[label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    return label_to_id
