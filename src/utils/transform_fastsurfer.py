import numpy as np
import nibabel as nib
from tools import list_dirs, list_files, glob_file
from pathlib import Path

FS_LABELS = [0, 2, 3, 41, 42]

GM_LABEL = 10
WM_LABEL = 99

FS_WM = [2, 42]
FS_GM = []


def get_paths_from_txt(txt):
    with open(txt) as f:
        paths = [line.rstrip() for line in f]
    return [Path(p) for p in paths]


def aparaseg_to_aseg(img):
    img[img >= 2000] = 42
    img[img >= 1000] = 3
    return img

def aseg_to_fs_gmwm(img):
    img = img.astype(int)
    new = np.zeros(img.shape, dtype=int)
    new = np.where(img==42, 42, new)
    new = np.where(img==3, 3, new)
    new = np.where(img==41, 41, new)
    new = np.where(img==2, 2, new)
    return new



def transform_fastsurfer_to_gmwm(input_txt, output_path):
    subjects = get_paths_from_txt(input_txt)
    for sub in subjects:

        sub_path = Path(sub)

        sub_obj = nib.load(sub_path)
        sub_img = sub_obj.get_fdata()

        new_img = aparaseg_to_aseg(sub_img)
        new_img = aseg_to_fs_gmwm(new_img)

        new_sub = nib.nifti1.Nifti1Image(new_img.astype(np.int32), affine=sub_obj.affine, header=sub_obj.header)
        print(f"saving {sub_path.name}")

        nib.save(new_sub, Path.joinpath(Path(output_path), sub_path.name))




if __name__ == "__main__":
    transform_fastsurfer_to_gmwm("/home/lmahler/code/mpi_val/src/utils/fastsurfer_seg.txt", "/home/lmahler/data/scratch/fastsurfer_val/out/")
