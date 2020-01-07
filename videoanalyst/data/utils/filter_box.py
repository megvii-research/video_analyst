
def filter_unreasonable_training_boxes(im, bbox, config):
    """ filter to small,too large or ratio far from  objects """

    # target_image, target_bbox, search_image, search_bbox, distractor_boxes, neg_pair = anno

    eps = 1e-6
    im_area = im.shape[0] * im.shape[1]
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    bbox_area_rate = bbox_area / im_area
    bbox_ratio = (bbox[3]-bbox[1]+1) / max(bbox[2]-bbox[0]+1, eps)

    # valid trainng box condition
    conds = [(config.min_area_rate < bbox_area_rate < config.max_area_rate),
             max(bbox_ratio, 1.0 / max(bbox_ratio, eps)) < config.max_ratio]

    # print("bbox_area_rate", bbox_area_rate)
    # from IPython import embed;embed()

    # if not all conditions are satisfied, filter the box
    return not all(conds)
