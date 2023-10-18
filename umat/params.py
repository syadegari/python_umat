import os
import yaml
import argparse
import logging
from typing import Union
import torch

from dataclasses import dataclass


PARAMS_SCHEMA = PARAMS_SCHEMA = {
    # weights for the loss
    "coeff_loss_data": {
        "type": float,
        "help": "Coefficient for the data loss.",
    },
    "coeff_loss_physics": {
        "type": float,
        "help": "Coefficient for the physics-based loss.",
    },
    "coeff_penalty_delta_gamma": {
        "type": float,
        "help": "Coefficient for the delta gamma penalty.",
    },
    "coeff_penalty_positive_gamma": {
        "type": float,
        "help": "Coefficient for the positive gamma penalty.",
    },
    "coeff_penalty_min_slipres": {
        "type": float,
        "help": "Coefficient for the minimum slip resistance penalty.",
    },
    "coeff_penalty_max_slipres": {
        "type": float,
        "help": "Coefficient for the maximum slip resistance penalty.",
    },
    # norm of penalty error
    "penalty_pnorm": {
        "type": float,
        "help": "P-norm value for the penalty error.",
    },
    # logging
    "log_flags": {
        "type": str,
        "choices": ["loss", "loss_grad_norm", "params_histogram"],
        "nargs": "*",
        "help": (
            "List of logs to be captured. Choose from 'loss', 'loss_grad_norm',"
            " or 'params_histogram'."
        ),
    },
    "log_frequencies": {
        "type": int,
        "nargs": "*",
        "help": "List of frequencies corresponding to each log specified in 'log_flags'.",
    },
    "tboard_path": {"type": str, "help": "Path for saving TensorBoard logs."},
    "sims_path": {"type": str, "help": "Path to the simulations."},
    "n_batch": {"type": int, "help": "Batch size for training."},
    "n_epoch": {"type": int, "help": "Number of epochs for training."},
    "opt_lr": {"type": float, "help": "Learning rate for the optimizer."},
    # Uncomment these if needed and provided with help descriptions
    # "opt_name": {
    #     "type": str,
    #     "choices": ["SGD", "Adam", "AdamW"],
    #     "help": "Name of the optimizer to use."
    # },
    # "opt_lr_decay_name": {
    #     "type": str,
    #     "choices": ["Linear"],
    #     "help": "Name of the learning rate decay method to use."
    # },
    # "opt_lr_decay_rate": {
    #     "type": float,
    #     "help": "Decay rate for the learning rate if using a decay method."
    # },
    "generate_config": {
        "action": "store_true",
        "default": None,
        "help": "Generates the modified config file and quits.",
    },
    "experiment_name": {"type": str, "help": "Name of the experiment."},
}


def colorize(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"


TYPE_COLOR_MAPPING = {
    str: colorize("STR", "94"),  # Light blue
    float: colorize("FLOAT", "92"),  # Light green
    int: colorize("INT", "93"),  # Yellow
    bool: colorize("BOOL", "95"),  # Light purple
}
#
for key, value in PARAMS_SCHEMA.items():
    if "type" in value:
        value["metavar"] = TYPE_COLOR_MAPPING.get(value["type"], value["type"])


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter):
    def __init__(
        self,
        prog: str,
        indent_increment: int = 2,
        max_help_position: int = 50,
        width: Union[int, None] = None,
    ) -> None:
        super().__init__(prog, indent_increment, max_help_position, width)


class MissingKeysError(Exception):
    pass


class WrongValueError(Exception):
    pass


def check_params_values(params):
    """
    parameters with choices should have valid subset of allowable values
    """
    for param_name in params:
        if "choices" in PARAMS_SCHEMA[param_name]:
            if not set(params[param_name]) <= set(PARAMS_SCHEMA[param_name]["choices"]):
                logging.error(f"Wrong value for {param_name}")
                raise WrongValueError(f"Wrong value for {param_name}: {params[param_name]}")


def check_params_exist(params):
    """
    All the parameters in the SCHEMA should be specified at this point.
    """
    missing_keys = PARAMS_SCHEMA.keys() - params.keys()
    if missing_keys:
        logging.error(f"Missing keys in config: {missing_keys}")
        raise MissingKeysError(missing_keys)
    else:
        return


def check_logging_info(params):
    """
    Since for the logging we specify logging type and logging frequency,
    we check that all the specified loggings have frequency
    """
    assert len(params["log_flags"]) == len(
        params["log_frequencies"]
    ), "the length of items in log_flags and log_frequencies should be equal"


def get_params(config_path, cmdline_params):
    """
    - Read the parameters from the cmdline.
    - Read parameters from the config file.
    - Overwrite the parameters from the cmdline to the config file.
    """
    # remove None values from cmdline arguments
    cmdline_params = {k: v for k, v in cmdline_params.items() if v is not None}
    rewrite_config_file = False
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)
    try:
        # replace the specified parameters from the cmdline in params dict
        for param_name, param_val in cmdline_params.items():
            if param_name != "config_file":
                if param_val is not None:
                    rewrite_config_file = True
                    params[param_name] = param_val
        #
        check_params_exist(params)
        check_params_values(params)
        check_logging_info(params)
        if rewrite_config_file:
            modified_config_file = (
                f"{os.path.dirname(config_path)}/config_{params['experiment_name']}.yaml"
            )
            with open(modified_config_file, "w") as f:
                f.writelines(yaml.dump(params))
        if params["generate_config"]:
            exit()
        return params
    #
    except MissingKeysError as k:
        logging.error(f"The followins are missing from {config_path}: {k}")
        raise
    except WrongValueError as w:
        logging.error(f"The following values are incorrect :{w}")
        raise


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
