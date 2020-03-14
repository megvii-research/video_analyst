# -*- coding: utf-8 -*
from paths import ROOT_PATH  # isort:skip
import demo
from demo.resources.static_img_example.getImage import im, bbox, im_z, im_x, target_bbox, search_bbox

from copy import deepcopy
import cv2
import logging

logger = logging.getLogger('global')

color = dict(
    target=(0, 255, 0),
    pred=(0, 255, 255),
    template=(0, 127, 255),
    search=(255, 0, 0),
)

if __name__ == "__main__":
    im_ = deepcopy(im)
    cv2.rectangle(im_, bbox[:2], bbox[2:], color["target"])
    cv2.rectangle(im_, target_bbox[:2], target_bbox[2:], color["template"])
    cv2.rectangle(im_, search_bbox[:2], search_bbox[2:], color["search"])
    font_size = 0.5
    font_width = 1
    cv2.putText(im_, "target box", (20, 20), cv2.FONT_HERSHEY_COMPLEX,
                font_size, color["target"], font_width)
    cv2.putText(im_, "template patch region", (20, 40),
                cv2.FONT_HERSHEY_COMPLEX, font_size, color["template"],
                font_width)
    cv2.putText(im_, "search patch region", (20, 60), cv2.FONT_HERSHEY_COMPLEX,
                font_size, color["search"], font_width)

    cv2.imshow("im_", im_)

    cv2.imshow("template", im_z)
    cv2.imshow("search", im_x)
    cv2.waitKey(0)
