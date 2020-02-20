# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip

import argparse
import logging
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

cv2.setNumThreads(1)

# torch.backends.cudnn.enabled = False

logger = logging.getLogger('global')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--config',
                        default='',
                        type=str,
                        help='path to experiment configuration')
    parser.add_argument('--resume-from-epoch',
                        default=-1,
                        type=int,
                        help=r"latest completed epoch's number (from which training resumes)")
    parser.add_argument('--resume-from-file',
                        default="",
                        type=str,
                        help=r"latest completed epoch's snapshot file (from which training resumes)")

    return parser

if __name__ == '__main__':
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()
    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)
    logger.info("Merged with root_cfg imported from videoanalyst.config.config.cfg")
    # resolve config
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    root_cfg = root_cfg.train
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()
    # backup config
    cfg_bak_dir = osp.join(task_cfg.exp_save, task_cfg.exp_name, "logs")
    ensure_dir(cfg_bak_dir)
    cfg_bak_file = osp.join(cfg_bak_dir, "%s_bak.yaml"%task_cfg.exp_name)
    with open(cfg_bak_file, "w") as f:
        f.write(task_cfg.dump())
    logger.info("Task configuration backed up at %s"%cfg_bak_file)
    # build model
    model = model_builder.build(task, task_cfg.model)
    # load data
    with Timer(name="Dataloader building", verbose=True, logger=logger):
        dataloader = dataloader_builder.build(task, task_cfg.data)
    # build optimizer
    optimizer = optim_builder.build(task, task_cfg.optim, model)
    # build trainer
    trainer = engine_builder.build(task, task_cfg.trainer, "trainer", optimizer,
                                   dataloader)
    trainer.resume(parsed_args.resume_from_epoch, parsed_args.resume_from_file)
    # trainer.init_train()
    logger.info("Start training")
    while not trainer.is_completed():
        trainer.train()
        trainer.save_snapshot()
    logger.info("Training completed.")
