import pandas as pd
import json
import yaml
import os
from tqdm import tqdm
from glob import glob
from typing import Dict, List, Any, Optional
import warnings


ENCODER_NAMES = {
    "albert-base-v2": "albert",
    "dbmdz/bert-base-german-uncased": "bert-german",
    "textattack/bert-base-uncased-MNLI": "bert-mnli",
    "dslim/bert-base-NER-uncased": "bert-conll",
    "vblagoje/bert-english-uncased-finetuned-pos": "bert-pos",
    "csarron/bert-base-uncased-squad-v1": "bert-squad",
    "bert-base-uncased": "bert",
    "uklfr/gottbert-base": "gottbert",
    "roberta-base": "roberta",
    "SpanBERT/spanbert-base-cased": "spanbert",
    "xlm-roberta-base": "xlm",
    "xlnet-base-cased": "xlnet",
}

OUTDATED_ENCODER_NAMES = {
    "bert-base-cased": "bert-cased",
    "dslim/bert-base-NER": "bert-conll-cased",
    "deepset/bert-base-cased-squad2": "bert-squad-cased",
}

READOUT_NAMES = {
    "LogisticRegression": "LR",
    "NearestCentroid": "NC",
    "NearestInstance": "NN",
}


def parse_hydra_config(config_path: str) -> Optional[Dict[str, Any]]:
    """Read a hydra config file (given in .yaml) for one experiment and return parsed configuration in a dict.

    Args:
        path: The path of hydra configuration .yaml file for one experiment.
        e.g. `./multirun/2021-10-11/14-25-22/0/.hydra/config.yaml` if it comes from "multirun" or
        `./outputs/2021-09-20/23-49-02/.hydra/config.yaml` if it comes from "outputs".

    Returns:
        Config given as a dict, containing `dataset`, `encoder`, `readout`, `nways`, `kshots`, `label_column`;\n
        For contrastive learning, extra  hyperparameter information should be included, such as
        `lr`, `weight_decay` and `num_epochs`.
    """
    with open(config_path, "r") as fp:
        dict_from_yaml = yaml.safe_load(fp)

    dataset = str(dict_from_yaml["dataset"]["path"])
    readout = str(dict_from_yaml["evaluation"]["classifier"]["_target_"])
    nways = int(dict_from_yaml["evaluation"]["dataset"]["n_ways"])
    kshots = int(dict_from_yaml["evaluation"]["dataset"]["k_shots"])
    label_column = str(dict_from_yaml["label_column_name"])

    dataset = dataset.strip("/").split("/")[-1].split(".")[0]
    readout = READOUT_NAMES[readout.split(".")[-1]]

    if dict_from_yaml["encoder"]["_target_"].split(".")[-1] == "RandomEncoder":
        encoder = "random"
    # then PLMs or contrastive PLMs
    else:
        encoder = str(dict_from_yaml["encoder"]["model_name_or_path"])

        if encoder not in ENCODER_NAMES:
            return
        encoder = ENCODER_NAMES[encoder]
        if (
            dict_from_yaml["encoder"]["_target_"].split(".")[-1]
            == "ContrastiveTransformerEncoder"
        ):
            encoder = "contrastive_" + encoder

    config = {
        "dataset": dataset,
        "encoder": encoder,
        "readout": readout,
        "nways": nways,
        "kshots": kshots,
        "label_column": label_column,
    }

    # collect hyperparameters for contrastive learning
    if encoder.split("_")[0] == "contrastive":
        config["num_epochs"] = int(dict_from_yaml["num_epochs"])
        config["lr"] = float(dict_from_yaml["learning_rate"])
        config["weight_decay"] = float(dict_from_yaml["weight_decay"])

    return config


def parse_json_result(result_path: str) -> Optional[Dict[str, Any]]:
    """Read a result file (given in .json) for one experiment and return parsed results in a dict.

    Args:
        path: The path of result .yaml file for one experiment.
        e.g. `./multirun/2021-10-11/14-25-22/0/evaluation_results.json` if it comes from "multirun" or
        `./outputs/2021-09-20/23-49-02/evaluation_results.json` if it comes from "outputs".

    Returns:
        Config given as a dict, containing `f1`, `f1_pm`
    """

    with open(result_path, "r") as fp:
        dict_from_json = json.load(fp)
    f1_micro_mean = dict_from_json["f1_micro"]["mean"]
    f1_micro_pm = dict_from_json["f1_micro"]["margin_of_error"]

    result = {
        "f1_micro": "{:.2f}".format(f1_micro_mean * 100),
        "f1_micro_pm": "{:.2f}".format(f1_micro_pm * 100),
    }
    return result


def logdir_to_dict(dir_path: str) -> Optional[Dict[str, Any]]:
    """Read a log directory for one experiment and return the configuration and result in a dict.

    Args:
        path: The path of log directory for one experiment.
        e.g. `./multirun/2021-10-11/14-25-22/0` if it comes from "multirun" or
        `./outputs/2021-09-20/23-49-02` if it comes from "outputs".

    Returns:
        A dictionary containing\n
            settings: `dataset`, `encoder`, `readout`, `nways`, `kshots`, `label_column`, `timestamp`;\n
            results: `f1`, `f1_pm`.\n
        For contrastive learning, extra  hyperparameter information should be included, such as
        `lr`, `weight_decay` and `num_epochs`.
    """
    result_path = os.path.join(dir_path, "evaluation_results.json")
    # only proceed if this experiment is completed
    if not os.path.exists(result_path):
        return

    # process config file
    config_path = os.path.join(dir_path, ".hydra/config.yaml")
    config = parse_hydra_config(config_path)

    # process result file
    result = parse_json_result(result_path)

    # fetch the timestamp
    dir_path_split = dir_path.strip("/").split("/")
    if dir_path_split[-4] == "multirun":
        date, time = dir_path_split[-3], dir_path_split[-2]
    elif dir_path_split[-3] == "outputs":
        date, time = dir_path_split[-2], dir_path_split[-1]
    timestamp = date + "/" + time

    # merge all the info into one dict
    if config is not None and result is not None:
        return {**config, **result, "timestamp": timestamp}


def scan_multirun(multirun_path: str = "./multirun"):  # -> List[Dict[str]]:
    """Scan all the experiments in folder `multirun`.

    Args:
        path: Path to multirun directory.

    Returns:
        A list of dictionaries, each corresponds to an experiment record.
    """
    log_dir_list = [
        log_dir
        for log_dir in list(glob("{}/*/*/*/".format(multirun_path)))
        if log_dir.split("/")[-1] != "multirun.yaml"
    ]

    records = []
    for log_dir in tqdm(log_dir_list):
        record = logdir_to_dict(log_dir)
        if record is not None:
            records.append(record)
    return records


def scan_outputs(outputs_path: str = "./outputs") -> List[Dict]:
    """Scan all the experiments in folder `multirun`.

    Args:
        path: Path to multirun directory.

    Returns:
        A list of dictionaries, each corresponds to an experiment record.
    """
    log_dir_list = [log_dir for log_dir in list(glob("{}/*/*/".format(outputs_path)))]

    records = []
    for log_dir in tqdm(log_dir_list):
        record = logdir_to_dict(log_dir)
        if record is not None:
            records.append(record)
    return records


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    records = scan_multirun() + scan_outputs()
    df = pd.DataFrame(records)
    print(df)

    print("reproduce Table 6:")
    new_df = df.loc[
        (df["dataset"].isin(["conll2003", "ontonotes"]))
        & (df["encoder"] == "albert")
        & (df["nways"] == 5)
    ]
    new_df.sort_values(["dataset", "kshots", "readout"], inplace=True)
    new_df.drop_duplicates(["dataset", "kshots", "readout"], keep="last", inplace=True)
    print(
        new_df[["dataset", "readout", "kshots", "f1_micro", "f1_micro_pm", "timestamp"]]
    )

    print("reproduce Table 5:")
    new_df = df.loc[
        (df["encoder"].isin(["bert", "bert-conll"]))
        & (df["nways"] == 5)
        & (df["readout"] == "LR")
    ]
    new_df.sort_values(["dataset", "label_column", "kshots", "encoder"], inplace=True)
    print(
        new_df[
            ["dataset", "label_column", "kshots", "encoder", "f1_micro", "f1_micro_pm"]
        ]
    )

    print("reproduce Table 4:")
    new_df = df.loc[
        (df["encoder"].isin(["bert", "bert-pos", "bert-mnli", "bert-squad"]))
        & (df["nways"] == 5)
        & (df["readout"] == "LR")
    ]
    new_df.sort_values(["dataset", "label_column", "kshots", "encoder"], inplace=True)
    new_df.drop_duplicates(["dataset", "label_column", "kshots", "encoder"], keep="last", inplace=True)
    print(
        new_df[
            ["dataset", "label_column", "kshots", "encoder", "f1_micro", "f1_micro_pm"]
        ]
    )

    print("reproduce Table 3:")
    new_df = df.loc[
        (df["encoder"].isin(["bert-german", "gottbert", "xlm"]))
        & (df["nways"] == 5)
        & (df["readout"] == "LR")
    ]
    new_df.sort_values(["dataset", "kshots", "encoder"], inplace=True)
    print(new_df[["dataset", "kshots", "encoder", "f1_micro", "f1_micro_pm"]])

    print("reproduce Table 2:")
    new_df = df.loc[
        (
            df["dataset"].isin(
                [
                    "conll2003",
                    "ontonotes",
                    "few-nerd",
                    "lenovo",
                    "wikiann",
                    "wnut_17",
                    "wikigold",
                ]
            )
        )
        & (
            df["encoder"].isin(
                ["random", "bert", "albert", "roberta", "spanbert", "xlnet"]
            )
        )
        & (df["nways"] == 5)
        & (df["readout"] == "LR")
    ]
    new_df.sort_values(["dataset", "label_column", "kshots", "encoder"], inplace=True)
    new_df.drop_duplicates(["dataset", "label_column", "kshots", "encoder"], keep="last", inplace=True)
    print(
        new_df[
            [
                "dataset",
                "label_column",
                "kshots",
                "encoder",
                "f1_micro",
                "f1_micro_pm",
                "timestamp",
            ]
        ]
    )
