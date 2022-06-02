import numpy as np
import nibabel as nib
from tools import list_dirs, list_files, glob_file
from pathlib import Path


GM_LABEL = 10
WM_LABEL = 99

FAST_GM= 2
FAST_WM = 3

def create_gmwm_img(img):
    img = img.astype(int)
    new = np.zeros(img.shape, dtype=int)
    new = np.where(img==FAST_GM, GM_LABEL, new)
    new = np.where(img==FAST_WM, WM_LABEL, new)
    return new

def get_paths_from_txt(txt):
    with open(txt) as f:
        paths = [line.rstrip() for line in f]
    return [Path(p) for p in paths]


def transform_ss_to_gmwm(in_txt, out_path):

    subjects = get_paths_from_txt(in_txt)

    for sub in subjects:

        sub_path = Path(sub)

        sub_obj = nib.load(sub_path)
        sub_img = sub_obj.get_fdata()

        sub_img = create_gmwm_img(sub_img)


        new_sub = nib.nifti1.Nifti1Image(sub_img, affine=sub_obj.affine, header=sub_obj.header)

        print(f"Saving {sub_path.name}")

        nib.save(new_sub, Path.joinpath(Path(out_path), sub_path.name))



if __name__ == "__main__":
    transform_ss_to_gmwm("/home/lmahler/code/mpi_val/src/utils/fast_seg.txt", "/home/lmahler/data/scratch/fast_val/")
