from typing import Optional, Union

import datasets

from genre.data.datasets import RE_DATASETS_ROOT


def load_dataset(
    dataset_name: str, split: str, version: Optional[str] = None, data_dir: Optional[str] = None
) -> Union[datasets.Dataset, datasets.DatasetDict]:
    dataset_script = (RE_DATASETS_ROOT / dataset_name).with_suffix(".py")
    return datasets.load_dataset(str(dataset_script), version, data_dir=data_dir, split=split)
