import os
import argparse
from umat.params import PARAMS_SCHEMA, HelpFormatter, get_params
from umat.train import train


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Command line arguments from schema",
        formatter_class=HelpFormatter,
    )
    parser.add_argument(
        "-c", "--config-file", type=str, help="Config file, including the path"
    )
    for key, value in PARAMS_SCHEMA.items():
        parser.add_argument(f"--{key}", **value)
    return parser.parse_args()


if __name__ == "__main__":
    #
    args = parse_arguments()
    params = get_params(os.path.abspath(args.config_file), vars(args))
    params["config_file_path"] = os.path.dirname(os.path.abspath(args.config_file))
    train(params)
