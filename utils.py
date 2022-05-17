import numpy as np


def transform_labels_to_fs(seg_arr, label_dict):
    """
    Transforms numpy array of segmentations to FreeSurfer labels

    Parameters
    ----------
    seg_arr: np.array
        array of labels
    label_dict: dictionary
        Maps labels of segmentation tool to FreeSurfer labels

    Returns
    -------
    res: np.array
        Transformed segmentation image with FreeSurfer labels
    """

    keys = np.array(list(label_dict.keys()))
    vals = np.array(list(label_dict.values()))
    res = np.zeros(keys.max() + 1, dtype=vals.dtype)
    res[keys] = vals
    return res[seg_arr]
