import numpy as np
import nibabel as nib
from tools import list_dirs, list_files, glob_file
from pathlib import Path


GM_LABEL = 10
WM_LABEL = 99

SS_GM = [3, 42]
SS_WM = [2, 41]

def create_gmwm_img(img):
    img = img.astype(int)
    new = np.zeros(img.shape, dtype=int)
    new = np.where(img==42, 42, new)
    new = np.where(img==3, 3, new)
    new = np.where(img==41, 41, new)
    new = np.where(img==2, 2, new)
    return new


def transform_ss_to_gmwm(base_path):
    subject_dirs = list_dirs(base_path)

    for sub in subject_dirs:


        files = list_files(sub)
        ss_path = glob_file(sub, "*_ss.nii")
        ss = nib.load(ss_path)
        
        s_name = ss_path.name.strip("_ss.nii")
        print("**********")
        print(f"Processing {s_name}...")
        
        ss_fast = glob_file(sub, "*_ss_fast.nii")
        ss_fast = nib.load(ss_fast)
       
        ss_rob = glob_file(sub, "*_ss_rob.nii")
        ss_rob = nib.load(ss_rob)

        ss_gmwm = create_gmwm_img(ss.get_fdata())
        ss_fast_gmwm = create_gmwm_img(ss_fast.get_fdata())
        ss_rob_gmwm = create_gmwm_img(ss_rob.get_fdata())


        ss_obj= nib.nifti1.Nifti1Image(ss_gmwm, affine=ss.affine, header=ss.header)
        ss_fast_obj= nib.nifti1.Nifti1Image(ss_fast_gmwm, affine=ss_fast.affine, header=ss_fast.header)
        ss_rob_obj= nib.nifti1.Nifti1Image(ss_rob_gmwm, affine=ss_rob.affine, header=ss_rob.header)

        nib.save(ss_obj, Path.joinpath(sub, s_name + "_ss_gmwm.nii"))
        nib.save(ss_fast_obj, Path.joinpath(sub, s_name + "_ss_fast_gmwm.nii"))
        nib.save(ss_rob_obj, Path.joinpath(sub, s_name + "_ss_rob_gmwm.nii"))



if __name__ == "__main__":
    transform_ss_to_gmwm("/home/lmahler/data/scratch/synthseg_out")
