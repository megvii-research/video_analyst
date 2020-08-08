# encoding: utf-8
# [Version 1] this is a archived version of densebox target maker
# current version is under _make_densebox_target.py_ (i.e. without "_v1" suffix)

import os
from typing import Dict, Tuple

import numpy as np

DUMP_FLAG = False  # dump intermediate results for debugging
DUMP_DIR = "dump"
DUMP_SUFFIX = "v1"
if not os.path.exists(DUMP_DIR):
    os.makedirs(DUMP_DIR)


def make_densebox_target(gt_boxes: np.array, config: Dict) -> Tuple:
    """ v1
    Model training target generation function for densebox

    Arguments
    ---------
    gt_boxes : np.array
        ground truth bounding boxes with class, shape=(N, 5), order=(x0, y0, x1, y1, class)
    config: configuration of target making (old format)
        Keys
        ----
        x_size : int
            search image size
        score_size : int
            score feature map size
        total_stride : int
            total stride of backbone
        score_offset : int
            offset between the edge of score map and the border of the search image

    Returns
    -------
    Tuple
        cls_res_final : np.array
            class
            shape=(N, 1)
        ctr_res_final : np.array
            shape=(N, 1)
        gt_boxes_res_final : np.array
            shape=(N, 4)
        # previous format
        # shape=(N, 6), order=(class, center-ness, left_offset, top_offset, right_offset, bottom_offset)
    """
    x_size = config["x_size"]
    score_size = config["score_size"]
    total_stride = config["total_stride"]
    score_offset = config["score_offset"]
    eps = 1e-5
    raw_height, raw_width = x_size, x_size

    # append class dimension to gt_boxes if ignored
    if gt_boxes.shape[1] == 4:
        gt_boxes = np.concatenate(
            [gt_boxes, np.ones(
                (gt_boxes.shape[0], 1))], axis=1)  # boxes_cnt x 5
    # l, t, r, b
    gt_boxes = np.concatenate([np.zeros((1, 5)), gt_boxes])  # (boxes_cnt, 5)
    gt_boxes_area = (np.abs(
        (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])))
    gt_boxes = gt_boxes[np.argsort(
        gt_boxes_area)]  # sort gt_boxes by area, ascending order
    boxes_cnt = len(gt_boxes)  # number of gt_boxes

    shift_x = np.arange(0, raw_width).reshape(-1, 1)
    shift_y = np.arange(0, raw_height).reshape(-1, 1)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # (H, W)

    # (H, W, #boxes, 1d-offset(l/t/r/b) )
    off_l = (shift_x[:, :, np.newaxis, np.newaxis] -
             gt_boxes[np.newaxis, np.newaxis, :, 0, np.newaxis])
    off_t = (shift_y[:, :, np.newaxis, np.newaxis] -
             gt_boxes[np.newaxis, np.newaxis, :, 1, np.newaxis])
    off_r = -(shift_x[:, :, np.newaxis, np.newaxis] -
              gt_boxes[np.newaxis, np.newaxis, :, 2, np.newaxis])
    off_b = -(shift_y[:, :, np.newaxis, np.newaxis] -
              gt_boxes[np.newaxis, np.newaxis, :, 3, np.newaxis])

    if DUMP_FLAG:
        off_l.dump("{}/off_l_{}.npz".format(DUMP_DIR, DUMP_SUFFIX))
        off_t.dump("{}/off_t_{}.npz".format(DUMP_DIR, DUMP_SUFFIX))
        off_r.dump("{}/off_r_{}.npz".format(DUMP_DIR, DUMP_SUFFIX))
        off_b.dump("{}/off_b_{}.npz".format(DUMP_DIR, DUMP_SUFFIX))

    # centerness
    center = ((np.minimum(off_l, off_r) * np.minimum(off_t, off_b)) /
              (np.maximum(off_l, off_r) * np.maximum(off_t, off_b) + eps))
    if DUMP_FLAG:
        center.dump("{}/center_{}.npz".format(DUMP_DIR, DUMP_SUFFIX))
    center = np.squeeze(np.sqrt(np.abs(center)))
    center[:, :, 0] = 0

    offset = np.concatenate([off_l, off_t, off_r, off_b],
                            axis=3)  # h x w x boxes_cnt * 4
    if DUMP_FLAG:
        offset.dump("{}/offset_{}.npz".format(DUMP_DIR, DUMP_SUFFIX))
    cls = gt_boxes[:, 4]

    cls_res_list = []
    ctr_res_list = []
    gt_boxes_res_list = []

    fm_height, fm_width = score_size, score_size

    fm_size_list = []
    fm_strides = [total_stride]
    fm_offsets = [score_offset]
    for fm_i in range(len(fm_strides)):
        fm_size_list.append([fm_height, fm_width])
        fm_height = int(np.ceil(fm_height / 2))
        fm_width = int(np.ceil(fm_width / 2))

    fm_size_list = fm_size_list[::-1]
    for fm_i, (stride, fm_offset) in enumerate(zip(fm_strides, fm_offsets)):
        fm_height = fm_size_list[fm_i][0]
        fm_width = fm_size_list[fm_i][1]

        shift_x = np.arange(0, fm_width)
        shift_y = np.arange(0, fm_height)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        xy = np.vstack(
            (shift_y.ravel(), shift_x.ravel())).transpose()  # (hxw) x 2
        # floor(stride / 2) + x * stride?
        off_xy = offset[fm_offset + xy[:, 0] * stride,
                        fm_offset + xy[:, 1] * stride]  # will reduce dim by 1
        # off_max_xy = off_xy.max(axis=2)  # max of l,t,r,b
        off_valid = np.zeros((fm_height, fm_width, boxes_cnt))

        is_in_boxes = (off_xy > 0).all(axis=2)
        # is_in_layer = (off_max_xy <=
        #         config.sep_win[fm_i]) & (off_max_xy >= config.sep_win[fm_i + 1])
        off_valid[
            xy[:, 0],
            xy[:,
               1], :] = is_in_boxes  #& is_in_layer  # xy[:, 0], xy[:, 1] reduce dim by 1 to match is_in_boxes.shape & is_in_layer.shape
        off_valid[:, :, 0] = 0  # h x w x boxes_cnt

        hit_gt_ind = np.argmax(off_valid, axis=2)  # h x w

        # gt_boxes
        gt_boxes_res = np.zeros((fm_height, fm_width, 4))
        gt_boxes_res[xy[:, 0],
                     xy[:, 1]] = gt_boxes[hit_gt_ind[xy[:, 0], xy[:, 1]], :4]
        gt_boxes_res_list.append(gt_boxes_res.reshape(-1, 4))

        # cls
        cls_res = np.zeros((fm_height, fm_width))
        cls_res[xy[:, 0], xy[:, 1]] = cls[hit_gt_ind[xy[:, 0], xy[:, 1]]]
        cls_res_list.append(cls_res.reshape(-1))

        # center
        center_res = np.zeros((fm_height, fm_width))
        center_res[xy[:, 0], xy[:, 1]] = center[fm_offset +
                                                xy[:, 0] * stride, fm_offset +
                                                xy[:, 1] * stride,
                                                hit_gt_ind[xy[:, 0], xy[:, 1]]]
        ctr_res_list.append(center_res.reshape(-1))
        # from IPython import embed;embed()

    cls_res_final = np.concatenate(cls_res_list,
                                   axis=0)[:, np.newaxis].astype(np.float32)
    ctr_res_final = np.concatenate(ctr_res_list,
                                   axis=0)[:, np.newaxis].astype(np.float32)
    gt_boxes_res_final = np.concatenate(gt_boxes_res_list,
                                        axis=0).astype(np.float32)

    # choose pos and neg point
    # labels = np.empty((len(cls_res_final),), dtype=np.float32)
    # labels.fill(-1)
    #
    # pos_index= np.where(cls_res_final > 0)
    # neg_index = np.where(cls_res_final == 0)
    # if len(pos_index[0]) > config.rpn_pos_samples:
    #     np.random.shuffle(pos_index[0])
    #     selected_pos = pos_index[0][:config.rpn_pos_samples]
    # else:
    #     selected_pos = pos_index[0]
    #
    # neg_num = config.rpn_total_samples - len(selected_pos)
    # np.random.shuffle(neg_index[0])
    # selected_neg = neg_index[0][:neg_num]
    #
    # labels[selected_pos] = 1
    # labels[selected_neg] = 0
    # labels = labels[:, np.newaxis]

    # return np.concatenate([cls_res_final, ctr_res_final, gt_boxes_res_final], axis=1)
    return cls_res_final, ctr_res_final, gt_boxes_res_final


if __name__ == '__main__':
    # gt_boxes
    gt_boxes = np.asarray([[13, 25, 100, 140, 1]])
    config_dict = dict(
        x_size=303,
        score_size=17,
        total_stride=8,
        score_offset=(303 - 1 - (17 - 1) * 8) // 2,
    )
    target = make_densebox_target(gt_boxes, config_dict)
    for v in target:
        print("{}".format(v.shape))
    from IPython import embed
    embed()
