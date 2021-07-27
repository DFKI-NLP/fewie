from typing import Dict, Optional, Union

import datasets
from fewie.data.datasets import NER_DATASETS_ROOT


def load_dataset(
    dataset_name: str, split: str, version: Optional[str] = None, data_dir: Optional[str] = None
) -> Union[datasets.Dataset, datasets.DatasetDict]:
    dataset_script = (NER_DATASETS_ROOT / dataset_name).with_suffix(".py")
    return datasets.load_dataset(str(dataset_script), version, data_dir=data_dir, split=split)


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def get_label_to_id(dataset, label_column_name: str) -> Union[Dict[str, int], Dict[int, int]]:
    if isinstance(dataset.features[label_column_name].feature, datasets.ClassLabel):
        label_list = dataset.features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(dataset[label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    return label_to_id
