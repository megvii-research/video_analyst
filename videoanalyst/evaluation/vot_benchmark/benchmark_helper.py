# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from os.path import join, realpath, dirname, exists, isdir
from os import listdir
import logging
import glob
import numpy as np
import json
import cv2
from collections import OrderedDict
import json
try:
    from neupeak.utils.brainpp.oss import OSSPath
    has_OSS = True
except:
    from pathlib import Path
    has_OSS = False

from pathlib import Path

from neupeak.utils import imgproc

_oss_prefix = "s3://"


def get_json(path, oss_file=True):
    # if oss_file:
    if oss_file and path.startswith(_oss_prefix):
        if not has_OSS:
            print('OSS not exists!')
            exit()
        if isinstance(path, str):
            p = OSSPath(path)
        else:
            p = path
        f = p.download()
        f = f.read()
        f = f.decode()
        f = json.loads(f)
        return f
    else:
        with open(path) as f:
            return json.load(f)


def get_txt(path, oss_file=True):
    # if oss_file:
    if oss_file and path.startswith(_oss_prefix):
        if not has_OSS:
            print('OSS not exists!')
            exit()
        if isinstance(path, str):
            p = OSSPath(path)
        else:
            p = path
        f = p.download()
        f = f.read()
        f = f.decode()
        return f

    else:
        with open(path) as f:
            return f.read()


def get_img(path, oss_file=True):
    # if oss_file:
    if oss_file and path.startswith(_oss_prefix):
        if not has_OSS:
            print('OSS not exists!')
            exit()
        if isinstance(path, str):
            p = OSSPath(path)
        else:
            p = path

        p = OSSPath(path)
        f = p.download().read()
        img = imgproc.imdecode(f)[:, :, :3]
    else:
        img = cv2.imread(path)
    return img


def get_files(path, suffix, oss_file):
    # if oss_file:
    if oss_file and path.startswith(_oss_prefix):
        if not has_OSS:
            print('OSS not exists!')
            exit()
        if isinstance(path, str):
            p = OSSPath(path)
        else:
            p = path
        list_dir = list(p.list_all())
    else:
        if isinstance(path, str):
            p = Path(path)
        else:
            p = path
        list_dir = list(p.glob('*'))
    result = [x.name for x in list_dir if x.suffix == suffix]
    return result


def get_dataset_zoo():
    root = realpath(join(dirname(__file__), '../data'))
    zoos = listdir(root)

    def valid(x):
        y = join(root, x)
        if not isdir(y): return False

        return exists(join(y, 'list.txt')) \
               or exists(join(y, 'train', 'meta.json'))\
               or exists(join(y, 'ImageSets', '2016', 'val.txt'))

    zoos = list(filter(valid, zoos))
    return zoos


#
# dataset_zoo = get_dataset_zoo()


def load_dataset(vot_path, dataset, oss_file=True):
    info = OrderedDict()
    if 'VOT' in dataset:
        base_path = join(vot_path, dataset)
        # if not exists(base_path):
        #     logging.error("Please download test dataset!!!")
        #     exit()
        list_path = join(base_path, 'list.txt')
        f = get_txt(list_path, oss_file)
        videos = [v.strip() for v in f.strip().split('\n')]
        #print(videos)
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, 'color')
            image_files = sorted(get_files(image_path, '.jpg', oss_file))
            image_files = [join(image_path, x) for x in image_files]
            gt_path = join(video_path, 'groundtruth.txt')
            gt = get_txt(gt_path, oss_file)
            gt = gt.strip().split('\n')

            gt = np.asarray([line.split(',') for line in gt], np.float32)

            if gt.shape[1] == 4:
                gt = np.column_stack(
                    (gt[:, 0], gt[:, 1], gt[:, 0], gt[:, 1] + gt[:, 3] - 1, gt[:, 0] + gt[:, 2] - 1,
                     gt[:, 1] + gt[:, 3] - 1, gt[:, 0] + gt[:, 2] - 1, gt[:, 1]))
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

    return info
