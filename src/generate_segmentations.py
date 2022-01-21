import argparse
import configparser


def parse_config(config_path):
    config = configparser.ConfigParser()
    cfg = config.read(config_path)
    return config

def run_all_segmentation_tools(config_path):
    # parse config file
    cfg = parse_config(config_path)
    for s in cfg.sections():
        print(s)

    # load paths

    # check if files exist

    # for each config entry run the run-command specified

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

    run_all_segmentation_tools(config_path=args.c)

