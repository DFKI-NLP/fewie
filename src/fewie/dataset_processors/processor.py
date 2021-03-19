from typing import List, Union

import datasets


class DatasetProcessor:
    def __call__(
        self, dataset: Union[datasets.Dataset, datasets.DatasetDict]
    ) -> Union[datasets.Dataset, datasets.DatasetDict]:
        pass
