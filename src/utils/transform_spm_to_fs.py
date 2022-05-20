import numpy as np
import nibabel as nib
from pathlib import Path

GM_LABEL = 10
WM_LABEL = 99

def list_dirs(base_path):
    subject_dirs = []
    for p in Path(base_path).iterdir():
        if p.is_dir():
            subject_dirs.append(p)
    return subject_dirs

def list_files(dir):
    return [x for x in dir.glob("**/*") if x.is_file()]

def glob_file(path, g):
    return [x for x in path.glob(g) if x.is_file()][0]

def threshold_img(img, threshold=0.5):
    return np.where(img > threshold, 1, 0)

def contruct_img(c1, c2):
    new = np.zeros(c1.shape)
    new = np.where(c1 == 1, GM_LABEL, new)
    new = np.where(c2 == 1, WM_LABEL, new)
    return new.astype(int)


def spm_to_fs_labels(base_path):
    subject_dirs = list_dirs(base_path)

    for sub in subject_dirs:
        files = list_files(sub)
        c1_path = glob_file(sub, "c1*")
        c2_path = glob_file(sub, "c2*")

        c1_obj = nib.load(c1_path)
        c2_obj = nib.load(c2_path)

        c1_bin_img = threshold_img(c1_obj.get_fdata())
        c2_bin_img = threshold_img(c2_obj.get_fdata())

        full_img = contruct_img(c1_bin_img, c2_bin_img)

        full_img_obj = nib.nifti1.Nifti1Image(full_img, affine=c1_obj.affine, header=c1_obj.header)
        new_name = c1_path.name.strip("c1").strip("_orig.nii") + "_full_spm.nii"
        nib.save(full_img_obj, Path.joinpath(sub, new_name))
        print(f"Saved {c1_path.name.strip('c1').strip('_orig.nii')}...")



if __name__ == "__main__":
    spm_to_fs_labels("/home/lmahler/data/scratch/spm_val")