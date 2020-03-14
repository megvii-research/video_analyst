# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip

import os
import argparse
import logging
import os.path as osp
import pickle

import cv2
from yacs.config import CfgNode

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

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

logger = logging.getLogger('global')


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
        default=-1,
        help=r"completed epoch's number, latest or one model path")

    return parser


def setup(rank: int, world_size: int):
    """Setting-up method to be called in the distributed function
       Borrowed from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    Parameters
    ----------
    rank : int
        process int
    world_size : int
        number of porocesses (of the process group)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group(
        "nccl", rank=rank,
        world_size=world_size)  # initialize the process group
    # torch.manual_seed(42)  # same initialized model for every process


def cleanup():
    """Cleanup distributed  
       Borrowed from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    """
    dist.destroy_process_group()


def run_dist_training(rank_id: int, world_size: int, task: str,
                      task_cfg: CfgNode, parsed_args, model):
    """method to run on distributed process
       passed to multiprocessing.spawn
    
    Parameters
    ----------
    rank_id : int
        rank id, ith spawned process 
    world_size : int
        total number of spawned process
    task : str
        task name (passed to builder)
    task_cfg : CfgNode
        task builder (passed to builder)
    parsed_args : [type]
        parsed arguments from command line
    """
    # set up distributed
    setup(rank_id, world_size)
    # build model
    # model = model_builder.build(task, task_cfg.model)
    # build optimizer
    optimizer = optim_builder.build(task, task_cfg.optim, model)
    # build dataloader with trainer
    with Timer(name="Dataloader building", verbose=True, logger=logger):
        dataloader = dataloader_builder.build(task, task_cfg.data, seed=rank_id)
    # build trainer
    trainer = engine_builder.build(task, task_cfg.trainer, "trainer", optimizer,
                                   dataloader)
    devs = ["cuda:%d" % rank_id]
    trainer.set_device(devs)
    trainer.resume(parsed_args.resume)
    # trainer.init_train()
    logger.info("Start training")
    while not trainer.is_completed():
        trainer.train()
        if rank_id == 0:
            trainer.save_snapshot()
        dist.barrier()  # one synchronization per epoch

    # clean up distributed
    cleanup()


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
    # build dummy dataloader (for dataset initialization)
    with Timer(name="Dummy dataloader building", verbose=True, logger=logger):
        dataloader = dataloader_builder.build(task, task_cfg.data)
    del dataloader
    logger.info("Dummy dataloader destroyed.")
    # build model
    model = model_builder.build(task, task_cfg.model)
    # prepare to spawn
    world_size = task_cfg.num_processes
    torch.multiprocessing.set_start_method('spawn', force=True)
    # spawn trainer process
    mp.spawn(run_dist_training,
             args=(world_size, task, task_cfg, parsed_args, model),
             nprocs=world_size,
             join=True)
    logger.info("Distributed training completed.")
