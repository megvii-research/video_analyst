from typing import Dict

import numpy as np

def filter_unreasonable_training_boxes(im: np.array, bbox, config: Dict) -> bool:
    r""" 
    Filter too small,too large objects and objects with extreme ratio

    Arguments
    ---------
    im: np.array
        image
    bbox: np.array or indexable object


    """
    eps = 1e-6
    im_area = im.shape[0] * im.shape[1]
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    bbox_area_rate = bbox_area / im_area
    bbox_ratio = (bbox[3]-bbox[1]+1) / max(bbox[2]-bbox[0]+1, eps)
    # valid trainng box condition
    conds = [(config["min_area_rate"] < bbox_area_rate, 
             bbox_area_rate < config["max_area_rate"]),
             max(bbox_ratio, 1.0 / max(bbox_ratio, eps)) < config["max_ratio"]
             ]
    # if not all conditions are satisfied, filter the box
    filter_flag = not all(conds)

    return filter_flag
