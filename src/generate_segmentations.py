import os
import argparse
import configparser
from pathlib import Path


def parse_config(config_path):
    """parse the config at the given path"""
    config = configparser.ConfigParser()
    cfg = config.read(config_path)
    return config

def check_or_make_dir(path):
    """checks if directory at given path exists, if not creates it"""
    Path(path).mkdir(parents=True, exist_ok=True)

def check_file(path):
    """check if file exists"""
    return os.path.isfile(path)

def sanitize_command(cmd):
    pass

def run_os_command(cmd):
    pass

def run_all_segmentation_tools(config_path):
    # parse config file
    cfg = parse_config(config_path)

    # load paths

    # check if files exist

    # for each config entry run the run-command specified

    for tool, cmd in cfg["TOOLS"].items():
        subprocess.run(cmd.split())


    # perform post segmentation actions

    #   combine labels
    #   correct labels
    #   convert to .nii
    pass


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Brain Segmentation Validation Framework")
    parser.add_argument("-c", type=str, help="Path to the config file")
    args = parser.parse_args()
    if args.c:
        run_all_segmentation_tools(config_path=args.c)
    else:
        print("Config not specified!")

