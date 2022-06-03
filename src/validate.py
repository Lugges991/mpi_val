import time
import nibabel as nib
import numpy as np
import argparse
import logging
from pathlib import Path
from nilearn.image import resample_img
import surface_distance
import pandas as pd
from tqdm import tqdm
from utils.tools import timeit, Timer

FS_LABELS = [0, 2, 3, 41, 42]
GM_LABEL = 10
WM_LABEL = 99


def get_paths_from_txt(txt):
    with open(txt) as f:
        paths = [line.rstrip() for line in f]
    return [Path(p) for p in paths]


def check_gt_seg_paths(gt, seg):

    if len(gt) != len(seg):
        logging.error("Segmentation and ground truth txt files do not contain the same number of entries")
        return False

    for g, s in zip(gt, seg):
        if not g.is_file():
            logging.error(f"{g} does not exist!")
            return False 

        if not s.is_file():
            logging.error(f"{s} does not exist!")
            return False 

    return True

def clean_labels(img):
    """ Replace all labels in img that are not in FS_LABELS with the background label 0"""
    img = img.astype(int)
    unq = np.unique(img)
    ind = np.isin(unq, FS_LABELS)
    for i, val in enumerate(ind):
        if not val:
            img = np.where(img == unq[i], 0, img)

    return img.astype(int)

def to_bool_label_arr(img):
    """ Transform img to binary label array where each dimension encodes one label"""
    unq = np.unique(img)
    res = []

    for i in unq:
        if i != 0:
            arr = np.zeros(img.shape)
            arr = np.where(img == i, 1, arr)
            res.append(arr)
    return np.array(res).astype(bool)

def fs_labels_to_gm_wm(img):
    img = np.where(img == 3, GM_LABEL, img)
    img = np.where(img == 42, GM_LABEL, img)
    
    img = np.where(img == 2, WM_LABEL, img)
    img = np.where(img == 41, WM_LABEL, img)

    return img


def run_evaluation(gt_txt, seg_txt, results_path="./results.csv", config=None):

    if config is not None:
        pass
    timer = Timer()
    gt_paths = get_paths_from_txt(gt_txt)
    seg_paths = get_paths_from_txt(seg_txt)

    if not check_gt_seg_paths(gt_paths, seg_paths):
        logging.error("Segmentation paths and ground truth paths mismatch")

    res = []

    for gt, seg in zip(gt_paths, seg_paths):

        print("**************")
        print(f"Processing subject: {gt.name.strip('_seg.nii')}")
        gt_obj= nib.load(gt)
        seg_obj= nib.load(seg)


        if gt_obj.shape != seg_obj.shape:
            timer.start()
            seg_obj = resample_img(seg_obj, target_affine=gt_obj.affine, target_shape=gt_obj.shape)
            timer.stop("resampling")
        gt_img = gt_obj.get_fdata()
        seg_img = seg_obj.get_fdata()

        print("Cleaning gt labels")
        gt_img = clean_labels(gt_img)
        gt_img = fs_labels_to_gm_wm(gt_img)

        print("Cleaning seg labels")
        seg_img = clean_labels(seg_img)
        seg_img = fs_labels_to_gm_wm(seg_img)
        
        print("Computing dice coefficient")
        dice = surface_distance.compute_dice_coefficient(gt_img, seg_img)

        gt_bins = to_bool_label_arr(gt_img)
        seg_bins = to_bool_label_arr(seg_img)

        sds = []
        asds = []
        hd95s = []
        
        for i in range(gt_bins.shape[0]):
            timer.start()
            print(f"Processing label {i} out of {len(gt_bins.shape[0])}")
            sd = surface_distance.compute_surface_distances(gt_bins[i], seg_bins[i], [1,1,1])
            sds.append(sd)
            asds.append(surface_distance.compute_average_surface_distance(sd))
            hd95s.append(surface_distance.compute_robust_hausdorff(sd, 95.))
            timer.stop(f"iteration {i}")
           
        res_dic = {"Name": gt.name.strip("_seg.nii"), "Dice": dice, "AverageSurfaceDistance": asds, "Hausdorff": hd95s}
        res.append(res_dic)

    res_df = pd.DataFrame(res)
    res_df.to_csv(results_path)





if __name__ == "__main__":

    # use config file instead:
    #     give gt.txt and seg.txt path in config
    #     also specify metrics in config


    parser = argparse.ArgumentParser(description="Evaluation of Segmentation Tools")
    parser.add_argument("--seg", help="Text file with paths to segmentation files")
    parser.add_argument("--gt", help="Text file with paths to ground truth labels. Have to correspond to segmentation paths in --seg")
    parser.add_argument("--res", help="Path where the results should be stored")
    args = parser.parse_args()

    run_evaluation(args.gt, args.seg, args.res)
