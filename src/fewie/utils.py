from omegaconf import DictConfig
import os


def resolve_relative_path(cfg: DictConfig, start_path: str):
    """Resolve all the relative paths given in `config.dataset` into absolute path.

    Args:
        cfg: Configuration of the experiment given in a dict.
        start_path: the absolute path of the starting point, usually the running \
            script that call this function.
    """
    # go from `start_path` up to the `fewie` project directory (i.e. `base_path`)
    base_path = start_path
    while os.path.dirname(base_path) not in ["/", ""]:
        if base_path[-6:] == "/fewie":
            break
        base_path = os.path.dirname(base_path)

    for config_column_name in ["path", "data_files"]:
        if config_column_name in cfg.dataset:
            path = cfg.dataset[config_column_name]
            # if the path is local relative
            if path[0] == ".":
                absolute_path = os.path.abspath(os.path.join(base_path, path))
                if not os.path.exists(absolute_path):
                    raise ValueError(
                        "Resolved absolute path {} does not exist, "
                        "please check your config path again".format(absolute_path))
                cfg.dataset[config_column_name] = absolute_path
