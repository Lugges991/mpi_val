# Brain segmentation validation framework

Contents:
This repo is organized in two parts.

- First, a script that automatically generates gray matter / white matter segmentations of given T1 images using a number of given software tools.
- Second, a framework to automatically validate, compare and visualize the different resulting segmentations with respect to ground truth manually labeled GMWM Segmentations.

The used segmentation tools are the following:

- [SPM](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)
- [FSL Fast](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FAST)
- [ANT](http://stnava.github.io/ANTs/)
- [MALP-EM](https://biomedia.doc.ic.ac.uk/software/malp-em/)
- [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)
- [FastSurfer](https://deep-mi.org/research/fastsurfer/)


Approach:
Skull stripping is used by each of the tools at some stage in their processing pipeline.
Thus skullstripping will be performed by each method on the native acquired image respectively.
As we aim to compare the segmentation quality of the aforementioned tools, we will perform the segmentation following the standard procedure as detailed by the respective documentation.

Expected Directory Structure:

data
└── SUBJECT1
    ├── SUBJECT1_raw
    │   └── SUBJECT1_T1.nii
    └── SUBJECT1_seg
        └── SUBJECT1_gmwm.nii


