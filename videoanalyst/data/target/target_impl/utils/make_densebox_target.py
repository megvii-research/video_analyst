# encoding: utf-8
from typing import Dict, Tuple

import numpy as np
import torch

DUMP_FLAG = False

def make_densebox_target(gt_boxes: np.array, config: Dict) -> Tuple:
    """
    Model training target generation function for densebox
    Only one resolution layer is taken into consideration

    About alignmenet with previous version
    - classification target: aligned
    - centerness target: slightly differ, 
                           e.g. 
                             max_err ~= 1e-8 in final centerness
                             max_err ~= 1e-6 in dense centerness
                         May due to the difference in implementation
                         of math operation (e.g. division)
    - bbox target: aligned

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

    gt_boxes = torch.from_numpy(gt_boxes).type(torch.FloatTensor)
    # gt box area
    #   TODO: consider change to max - min + 1?
    # (#boxes, 4-d_box + 1-d_cls)
    #   append dummy box (0, 0, 0, 0) at first for convenient
    #   #boxes++
    gt_boxes = torch.cat([torch.zeros(1, 5), gt_boxes], dim=0)
    # from IPython import embed;embed()

    gt_boxes_area = (torch.abs(
        (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])))
    # sort gt_boxes by area, ascending order
    #   small box priviledged to large box
    gt_boxes = gt_boxes[torch.argsort(gt_boxes_area)]
    # number of gt_boxes
    boxes_cnt = len(gt_boxes)

    x_coords = torch.arange(0, raw_width)  # (W, )
    y_coords = torch.arange(0, raw_height)  # (H, )
    y_coords, x_coords = torch.meshgrid(x_coords, y_coords)  # (H, W)

    off_l = (x_coords[:, :, np.newaxis, np.newaxis] -
             gt_boxes[np.newaxis, np.newaxis, :, 0, np.newaxis])
    off_t = (y_coords[:, :, np.newaxis, np.newaxis] -
             gt_boxes[np.newaxis, np.newaxis, :, 1, np.newaxis])
    off_r = -(x_coords[:, :, np.newaxis, np.newaxis] -
              gt_boxes[np.newaxis, np.newaxis, :, 2, np.newaxis])
    off_b = -(y_coords[:, :, np.newaxis, np.newaxis] -
              gt_boxes[np.newaxis, np.newaxis, :, 3, np.newaxis])

    if DUMP_FLAG:
        off_l.numpy().dump("off_l_new.npz")
        off_t.numpy().dump("off_t_new.npz")
        off_r.numpy().dump("off_r_new.npz")
        off_b.numpy().dump("off_b_new.npz")
    
    # centerness
    # (H, W, #boxes, 1-d_centerness)
    center = ((torch.min(off_l, off_r) * torch.min(off_t, off_b)) /
              (torch.max(off_l, off_r) * torch.max(off_t, off_b) + eps))
    # center = ((torch.min(off_l, off_r) * torch.min(off_t, off_b)) /
    #           torch.clamp(torch.max(off_l, off_r) * torch.max(off_t, off_b), min=eps))
    if DUMP_FLAG:
        center.numpy().dump("center_new.npz")
    # (H, W, #boxes, )
    center = torch.squeeze(torch.sqrt(torch.abs(center)), dim=3)
    center[:, :, 0] = 0  # mask centerness for dummy box as zero

    # (H, W, #boxes, 4)
    offset = torch.cat([off_l, off_t, off_r, off_b],
                            dim=3)
    if DUMP_FLAG:
        offset.numpy().dump("offset_new.npz")

    # (#boxes, )
    #   store cls index of each box
    #   class 0 is background
    #   dummy box assigned as 0
    cls = gt_boxes[:, 4]



    fm_height, fm_width = score_size, score_size  # h, w
    fm_offset = score_offset
    stride = total_stride
    x_coords = torch.arange(0, fm_width)  # (w, )
    y_coords = torch.arange(0, fm_height)  # (h, )
    y_coords, x_coords = torch.meshgrid(x_coords, y_coords)  # (H, W)
    y_coords = y_coords.reshape(-1)  # (hxw, ), flattened
    x_coords = x_coords.reshape(-1)  # (hxw, ), flattened


    # (hxw, #boxes, 4-d_offset_(l/t/r/b), )
    offset_on_fm = offset[fm_offset + y_coords * stride,
                    fm_offset + x_coords * stride]  # will reduce dim by 1
    # (hxw, #gt_boxes, )
    is_in_boxes = (offset_on_fm > 0).all(axis=2)
    # (h, w, #gt_boxes, ), boolean
    #   valid mask
    offset_valid = np.zeros((fm_height, fm_width, boxes_cnt))
    offset_valid[
        y_coords,
        x_coords, :] = is_in_boxes  #& is_in_layer  # xy[:, 0], xy[:, 1] reduce dim by 1 to match is_in_boxes.shape & is_in_layer.shape
    offset_valid[:, :, 0] = 0  # h x w x boxes_cnt
    
    # (h, w), boolean
    #   index of pixel on feature map
    #     used for indexing on gt_boxes, cls
    #   if not match any box, fall on dummy box at index 0
    #   if conflict, choose box with smaller index
    #   P.S. boxes already ordered by box's area
    hit_gt_ind = np.argmax(offset_valid, axis=2)

    # (h, w, 4-d_box)
    #   gt_boxes
    gt_boxes_res = torch.zeros((fm_height, fm_width, 4))
    gt_boxes_res[y_coords,
                 x_coords] = gt_boxes[hit_gt_ind[y_coords, x_coords], :4]  # gt_boxes: (#boxes, 5)
    gt_boxes_res = gt_boxes_res.reshape(-1, 4)
    # gt_boxes_res_list.append(gt_boxes_res.reshape(-1, 4))

    # (h, w, 1-d_cls_score)
    cls_res = torch.zeros((fm_height, fm_width))
    cls_res[y_coords, x_coords] = cls[hit_gt_ind[y_coords, x_coords]]
    cls_res = cls_res.reshape(-1, 1)

    # (h, w, 1-d_centerness)
    center_res = torch.zeros((fm_height, fm_width))
    center_res[y_coords, x_coords] = center[fm_offset + y_coords * stride, 
                                            fm_offset + x_coords * stride,
                                            hit_gt_ind[y_coords, x_coords]]
    center_res = center_res.reshape(-1, 1)


    return cls_res, center_res, gt_boxes_res


if __name__ == '__main__':
    # gt_boxes
    gt_boxes = np.asarray([[13, 25, 100, 140, 1]])
    config_dict = dict(
        x_size=303,
        score_size=17,
        total_stride=8,
        score_offset=(303-1 - (17-1)*8) // 2,
    )
    target = make_densebox_target(gt_boxes, config_dict)
    for v in target:
        print("{}".format(v.shape))
    from IPython import embed;embed()
