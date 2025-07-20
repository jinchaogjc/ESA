import copy
import os
import numpy as np
import numpy.ma as ma
# from PIL import Image
# from utils import resize_cv_img
from scipy.ndimage import zoom
import time

def get_proposals_label_with_count(active_proposal, gt):
    gt_cp = copy.copy(gt)
    gt_masked = ma.masked_array(gt_cp, mask=active_proposal)
    gt_masked[~gt_masked.mask] = 255

    count_gt_mask = gt_masked.data.reshape(-1)
    labels_in_es, counts = np.unique(count_gt_mask, return_counts=True)
    # print(label_counts)
    if labels_in_es[np.argmax(counts)]==255 and len(counts)>1:
        idx = -2
    else:
        idx = -1
    label = labels_in_es[np.argsort(counts)[idx]]
    
    active_proposal[active_proposal==1] = label
    del gt_masked
    return active_proposal, label


def enquiry_labels_from_gt(gt_labels, query_labels, dataset="voc"):
    # vocsbd_gt_filename = os.path.join(gt_dir, os.path.basename(gtfilename).split(".")[0] + ".png")
    gt = np.array(gt_labels[0])
    # print(np.unique(query_labels))
    count_list = list(np.unique(query_labels))
    active_mask = np.zeros(gt_labels.shape[1:], dtype=np.uint8)
    count = 1
    
    for idx in range(query_labels.shape[0]):
        active_mask_tmp = np.zeros(gt_labels.shape[1:], dtype=np.uint8)
        if dataset == "voc":
            scale_x, scale_y = gt_labels.shape[1]/query_labels[idx].shape[0], gt_labels.shape[2]/query_labels[idx].shape[1]
            query_label = zoom(query_labels[idx], (scale_x, scale_y), mode='nearest')
        elif dataset == "cityscapes":
            query_label = query_labels[idx]
        active_mask_tmp[query_label==count_list[count]] = 1
        active_mask_tmp, label = get_proposals_label_with_count(active_mask_tmp, gt)
        if label == 255:
            active_mask[query_label==count_list[count]] = 255
        elif label == 0:
            pass
        else:
            active_mask[query_label==count_list[count]] = label
        count += 1

    # if len(np.unique(active_mask)) > 2:
    #     print(np.unique(active_mask))
    return active_mask