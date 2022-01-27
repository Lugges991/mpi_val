import os
import argparse
import subprocess
import configparser
import time
import logging
from pathlib import Path

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

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

def list_dirs(path):
    """returns a list of all non-empty directories in the given path"""
    paths_list = []
    for p in Path(path).iterdir():
        if p.is_dir() and len(list(p.iterdir())) != 0:
            paths_list.append(p)
    return paths_list

def get_original_image(path, file_ending="*_orig.nii"):
    for p in Path(path).iterdir():
        if p.match(file_ending) is not None:
            return p

def check_or_make_dir(path):
    Path.mkdir(path, parents=True, exist_ok=True)

def run_all_segmentation_tools(config_path):
    # parse config file
    cfg = parse_config(config_path)

    # get list of paths of all subject directories
    base_path = cfg["PATHS"].get("base_subject_dir")
    subjects = list_dirs(base_path)
    # out_dir = Path(cfg["PATHS"].get("output_dir")).joinpath(int(time.time())) # for production
    out_dir = Path(cfg["PATHS"].get("output_dir")).joinpath("test")
    check_or_make_dir(out_dir)
    logging.info(f"Writing segmentations to {out_dir}")



    # iterate over all subjects
    for sub in subjects:
        # for each subject get the raw nifti file from 
        orig_img = get_original_image(sub)
        # create folder for each subject
        s_out_path= out_dir.joinpath(sub.name)
        check_or_make_dir(s_out_path)
        print(s_out_path)

    # for each config entry run the run-command specified
    # for tool, cmd in cfg["TOOLS"].items():
    #     subprocess.run(cmd.split())


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

