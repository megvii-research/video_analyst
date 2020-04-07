# -*- coding: utf-8 -*-
import os.path as osp

import cv2
import numpy as np
from loguru import logger
from PIL import Image

_RETRY_NUM = 10


def load_image(img_file: str) -> np.array:
    """Image loader used by data module (e.g. image sampler)
    
    Parameters
    ----------
    img_file: str
        path to image file
    Returns
    -------
    np.array
        loaded image
    
    Raises
    ------
    FileExistsError
        invalid image file
    RuntimeError
        unloadable image file
    """
    if not osp.isfile(img_file):
        logger.info("Image file %s does not exist." % img_file)
    # read with OpenCV
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if img is None:
        # retrying
        for ith in range(_RETRY_NUM):
            logger.info("cv2 retrying (counter: %d) to load image file: %s" %
                        (ith + 1, img_file))
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            if img is not None:
                break
    # read with PIL
    if img is None:
        logger.info("PIL used in loading image file: %s" % img_file)
        img = Image.open(img_file)
        img = np.array(img)
        img = img[:, :, [2, 1, 0]]  # RGB -> BGR
    if img is None:
        logger.info("Fail to load Image file %s" % img_file)

    return img
