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

cv2.setNumThreads(1)

# torch.backends.cudnn.enabled = False

# pytorch reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html#cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg',
                        '--config',
                        default='',
                        type=str,
                        help='path to experiment configuration')
    parser.add_argument(
        '-r',
        '--resume',
        default="",
        help=r"completed epoch's number, latest or one model path")

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
    task_cfg.freeze()
    # backup config
    cfg_bak_dir = osp.join(task_cfg.exp_save, task_cfg.exp_name, "logs")
    ensure_dir(cfg_bak_dir)
    cfg_bak_file = osp.join(cfg_bak_dir, "%s_bak.yaml" % task_cfg.exp_name)
    with open(cfg_bak_file, "w") as f:
        f.write(task_cfg.dump())
    logger.info("Task configuration backed up at %s" % cfg_bak_file)
    # log config
    logger.configure(handlers=[{"sink": sys.stderr, "level": 'INFO'}])
    logger.add(osp.join(cfg_bak_dir, "train_log.txt"),
               enqueue=True,
               backtrace=True,
               diagnose=True)
    logger.info("Task configuration backed up at %s" % cfg_bak_file)
    # device config
    if task_cfg.device == "cuda":
        world_size = task_cfg.num_processes
        assert torch.cuda.is_available(), "please check your devices"
        assert torch.cuda.device_count(
        ) >= world_size, "cuda device {} is less than {}".format(
            torch.cuda.device_count(), world_size)
        devs = ["cuda:{}".format(i) for i in range(world_size)]
    else:
        devs = ["cpu"]
    # build model
    model = model_builder.build(task, task_cfg.model)
    model.set_device(devs[0])
    # load data
    with Timer(name="Dataloader building", verbose=True):
        dataloader = dataloader_builder.build(task, task_cfg.data)
    # build optimizer
    optimizer = optim_builder.build(task, task_cfg.optim, model)
    # build trainer
    trainer = engine_builder.build(task, task_cfg.trainer, "trainer", optimizer,
                                   dataloader)
    trainer.set_device(devs)
    trainer.resume(parsed_args.resume)
    # trainer.init_train()
    logger.info("Start training")
    while not trainer.is_completed():
        trainer.train()
        trainer.save_snapshot()
    logger.info("Training completed.")
