# encoding: utf-8
import numpy as np
import cv2
from .bbox import cxywh2xyxy

def get_axis_aligned_bbox(region):
    """
    Get axis-aligned bbox (needed by VOT benchmark)
    :param region:
    :return:
    """
    try:
        region = np.array([region[0][0][0], region[0][0][1], region[0][1][0], region[0][1][1],
                           region[0][2][0], region[0][2][1], region[0][3][0], region[0][3][1]])
    except:
        region = np.array(region)
    cx = np.mean(region[0::2])
    cy = np.mean(region[1::2])
    x1 = min(region[0::2])
    x2 = max(region[0::2])
    y1 = min(region[1::2])
    y2 = max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
        np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1
    return cx, cy, w, h


def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans=(0, 0, 0)):
    """
    Get subwindow via cv2.warpAffine
    :param im: original image
    :param pos: subwindow position
    :param model_sz: output size
    :param original_sz: subwindow range on the original image
    :param avg_chans: average values per channel
    :return:
    """
    crop_cxywh = np.concatenate([np.array(pos), np.array((original_sz, original_sz))], axis=-1)
    crop_xyxy = cxywh2xyxy(crop_cxywh)
    # warpAffine transform matrix
    M_13 = crop_xyxy[0]
    M_23 = crop_xyxy[1]
    M_11 = (crop_xyxy[2]-M_13)/(model_sz-1)
    M_22 = (crop_xyxy[3]-M_23)/(model_sz-1)
    mat2x3 = np.array([M_11, 0   , M_13,
                       0   , M_22, M_23, ]).reshape(2, 3)
    im_patch = cv2.warpAffine(im, mat2x3, (model_sz, model_sz), flags=(cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(map(int, avg_chans)))
    return im_patch


def get_crop(im, target_pos, target_sz, z_size, x_size=None, avg_chans=(0, 0, 0), context_amount=0.5,
             func_get_subwindow=get_subwindow_tracking):
    """
    Get cropped patch for tracking
    :param im:
    :param target_pos:
    :param target_sz:
    :param z_size:
    :param x_size:
    :param avg_chans:
    :param context_amount:
    :param func_get_subwindow:
    :return:
    """
    wc = target_sz[0] + context_amount * sum(target_sz)
    hc = target_sz[1] + context_amount * sum(target_sz)
    s_crop = np.sqrt(wc * hc)
    scale = z_size / s_crop

    # im_pad = x_pad / scale
    # s_crop_aug = s_crop + 2 * im_pad
    if x_size is None:
        x_size = z_size
    s_crop = x_size / scale

    # x_size_aug = z_size + 2 * x_pad
    # extract scaled crops for search region x at previous target position
    im_crop = func_get_subwindow(im, target_pos, x_size, round(s_crop), avg_chans)

    return im_crop, scale
