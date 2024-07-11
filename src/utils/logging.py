import os
import sys
import pandas as pd
import torch
from typing import List


def store_labels(tensors: List[torch.tensor], columns: List[str],
                 output_dir: str, filename: str):
    """ Stores the labels in a csv file

    Args:
        tensors (List[torch.tensor]): list of tensors
        columns (List[str]): column names
        output_dir (str): output directory
        filename (str): name of the output file
    """    
    tensors = tensors[1:, :]
    np_array = tensors.cpu().numpy()
    df = pd.DataFrame(data=np_array, columns=columns)
    df.to_csv(os.path.join(output_dir, filename))


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
