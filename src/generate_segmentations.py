import os
import argparse
import subprocess
import configparser
import time
import logging
import glob
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
    origs = glob.glob(str(path.joinpath(file_ending)))
    return origs[0] if origs else None

def run_FAST():
    # run skullstrip

    # run segmentation
    subprocess.run(t_cmd.split())
    pass

def run_ANT():
    pass

def run_MALPEM():
    pass

def run_FASTSURFER():
    pass

def run_tool(tool_name, command, original_image_path, output_path):
    locals()[f"run_{tool_name}"]()

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
        # for each subject get the raw nifti file from the base path
        orig_img = get_original_image(sub)
        if orig_img:
            # create output folder for each subject
            s_out_path= out_dir.joinpath(sub.name)
            check_or_make_dir(s_out_path)

            # for each tool, run the segmentation
            print(20*"+")
            for tn, t_cmd in cfg["TOOLS"].items():
                t_s_path = s_out_path.joinpath(tn)
                check_or_make_dir(t_s_path)

                # replace <SUBJECT> placeholder with path to original img of subject
                t_cmd = t_cmd.replace("<SUBJECT>", str(orig_img))
                t_cmd = t_cmd.replace("<OUT>", str(t_s_path.joinpath(f"{sub.name}_seg")))
                t_cmd = t_cmd.replace("<SKULL>", str(t_s_path.joinpath(f"{sub.name}_skullstrip")))
                logging.info(t_cmd)
                # check if t_cmd has more than one bash command, if so, run both separately
                if ";" in t_cmd:
                    new_cmd = t_cmd.split(";")
                    logging.info(f"Running {tn} on {sub.name}")
                    for c in new_cmd:
                        logging.info(c)
                        subprocess.run(c.split())
                else:
                    # run_tool(tn, t_cmd, orig_img, t_s_path)
                    logging.info(f"Running {tn} on {sub.name}")
                    logging.info(t_cmd)
                    subprocess.run(t_cmd.split())


                # how do we handle intermediate outputs???
            break
        else:
            logging.info(f"No input image found for subject {sub.name}")



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

