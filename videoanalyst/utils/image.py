# -*- coding: utf-8 -*-
import os.path as osp
import logging

import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger("global")

def load_image(img_file: str) -> np.array:
    """Image loader used by data module (e.g. image sampler)
    
    Parameters
    ----------
    img_file : str
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
        raise FileExistsError("Image file %s does not exist."%img_file)
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if img is None:
        img = Image.open(img_file)
        img = np.array(img)
        img = img[:, :, [2,1,0]]  # RGB -> BGR
        logger.info("PIL used in loading image file: %s"%img_file)
    if img is None:
        raise RuntimeError("Fail to load Image file %s"%img_file)

    return img
