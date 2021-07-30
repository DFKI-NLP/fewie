from omegaconf import DictConfig
import os


def resolve_relative_path(cfg: DictConfig, base_path: str):
    """Resolve all the relative paths given in `config.dataset` into absolute path.

    Args:
        cfg: Configuration of the experiment given in a dict.
        base_path: the absolute path of the base directory.
    """
    for config_column_name in ["path", "data_files"]:
        path = cfg.dataset[config_column_name]
        # if the path is local relative
        if path[:2] in ["./", "../"]:
            absolute_path = os.path.abspath(os.path.join(base_path, "../", path))
            if not os.path.exists(absolute_path):
                raise ValueError(
                    "Resolved absolute path {} does not exist, "
                    "please check your config path again".format(absolute_path))
            cfg.dataset[config_column_name] = absolute_path
