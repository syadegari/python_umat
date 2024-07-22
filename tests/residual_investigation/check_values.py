import pandas as pd
from enum import Enum
import sys
from colorama import Fore

threshold_value: float = 1e-11
warning_threshold: int = 10  # Number of values exceeding threshold for warning


allowable_names = ["r_I norm", "r_II norm", "relative stress error"]


class Status(Enum):
    OK = f"{Fore.GREEN}OK{Fore.RESET}"
    Warning = f"{Fore.YELLOW }WARNING{Fore.RESET}"
    Failure = f"{Fore.RED}FAIL{Fore.RESET}"


def print_info():
    info = """
Checks the values in each column and print information if they are bigger than certain threshold.
OK: All values are below threshold.
Warning: Up to {num_threshold} values are bigger than the threshold
Failure: More than {num_threshold} are bigger than the threshold
The threshold is : {threshold}
    """.format(
        threshold=threshold_value, num_threshold=warning_threshold
    )
    print(info)


def check_values(filename):
    # Read the CSV file
    df = pd.read_csv(filename)

    print_info()

    for column in df.columns:
        column_name = column.strip()
        if column_name not in allowable_names:
            continue
        values = df[column]
        num_values = len(values)
        num_above_threshold = (values > threshold_value).sum()
        percent_above_threshold = (num_above_threshold / num_values) * 100

        if num_above_threshold == 0:
            status = Status.OK
        elif num_above_threshold <= warning_threshold:
            status = Status.Warning
        else:
            status = Status.Failure

        print(f"{column_name}: {status.value}")

        if status != Status.OK:
            violating_values = values[values > threshold_value]
            min_val = violating_values.min()
            max_val = violating_values.max()
            avg_val = violating_values.mean()

            print(f"    Percent of points above threshold: {percent_above_threshold:.2f}")
            print(f"    Number of points above threshold: {num_above_threshold}")
            print(f"    Min value above threshold: {min_val:.6e}")
            print(f"    Max value above threshold: {max_val:.6e}")
            print(f"    Average value above threshold: {avg_val:.6e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_values.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    check_values(filename)
