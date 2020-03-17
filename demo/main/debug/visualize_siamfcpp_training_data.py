# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip

import argparse
from loguru import logger
import os.path as osp
import pickle

import cv2

import torch

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.data import builder as dataloader_builder
from videoanalyst.engine import builder as engine_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.model.loss import builder as losses_builder
from videoanalyst.optim import builder as optim_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.utils import Timer, ensure_dir, complete_path_wt_root_in_cfg

from videoanalyst.data.utils.visualization import show_img_FCOS

cv2.setNumThreads(1)

# torch.backends.cudnn.enabled = False

# pytorch reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html#cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        '--config',
        default='experiments/siamfcpp/train/siamfcpp_alexnet-trn.yaml',
        type=str,
        help='path to experiment configuration')

    return parser


if __name__ == '__main__':
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()
    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)
    logger.info(
        "Merged with root_cfg imported from videoanalyst.config.config.cfg")
    # resolve config
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    root_cfg = root_cfg.train
    task, task_cfg = specify_task(root_cfg)
    task_cfg.data.num_workers = 1
    task_cfg.data.sampler.submodules.dataset.GOT10kDataset.check_integrity = False
    task_cfg.freeze()

    # load data
    with Timer(name="Dataloader building", verbose=True):
        dataloader = dataloader_builder.build(task, task_cfg.data)

    for batch_training_data in dataloader:
        keys = list(batch_training_data.keys())
        batch_size = len(batch_training_data[keys[0]])
        training_samples = [{
            k: v[[idx]]
            for k, v in batch_training_data.items()
        } for idx in range(batch_size)]
        for training_sample in training_samples:
            # from IPython import embed;embed()
            target_cfg = task_cfg.data.target
            show_img_FCOS(target_cfg[target_cfg.name], training_sample)
