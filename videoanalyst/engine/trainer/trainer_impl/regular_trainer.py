# -*- coding: utf-8 -*
import copy
import itertools
import logging
import math
import os
import os.path as osp
from collections import OrderedDict
from os.path import join

from typing import List

import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import Base

from ..trainer_base import TRACK_TRAINERS, TrainerBase
from videoanalyst.utils import ensure_dir, move_data_to_device

from videoanalyst.model.module_base import ModuleBase
from videoanalyst.optimizer.optimizer_base import OptimizerBase

logger = logging.getLogger("global")


@TRACK_TRAINERS.register
class RegularTrainer(TrainerBase):
    r"""
    Trainer to test the vot dataset, the result is saved as follows
    exp_dir/logs/$dataset_name$/$tracker_name$/baseline
                                    |-$video_name$/ floder of result files
                                    |-eval_result.csv evaluation result file

    Hyper-parameters
    ----------------
    devices: List[str]
        list of string
    num_iterations: int
        number of iterations
    """
    default_hyper_params = dict(
        exp_name="default_training",
        exp_save="snapshots",
        devices=["cpu"],
        num_iterations=1,
        max_epoch=1,
    )

    def __init__(self, 
                 model: ModuleBase, 
                 dataloader: DataLoader, 
                 losses: ModuleBase, 
                 optimizer: OptimizerBase, 
                 process=[]):
        r"""
        Crete tester with config and pipeline

        Arguments
        ---------
            model: ModuleBase,
            dataloder: DataLoader, 
            losses: ModuleBase,
            processes: ProcessBase

        """
        super(RegularTrainer, self).__init__()
        # famous four elements in Deep Laerning (c.f. <Deep Learning>, Goodfellow et al.)
        self.model = model
        self.dataloder = dataloder
        self.losses = losses
        self.optimizer = optimizer
        # update state
        self._state["epoch"] = 0
        self.update_params()

    def update_params(self, ):
        self._state["devices"] = [torch.device(dev) for dev in self._hyper_params["devices"]]
        self._state["snapshot_dir"] = osp.join(self._hyper_params["exp_save", 
                                               self._hyper_params["exp_name"])

    def init_train():
        self.model.train()
        self.model.cuda()
        torch.cuda.empty_cache()
        if len(self._state["devices"]) > 1:
            model = nn.DataParallel(model)
            logger.info("Use nn.DataParallel for data parallelism")

    
    def train(self):
        epoch = self._state["epoch"]
        max_epoch = self._hyper_params["max_epoch"]

        self.optimzier.schedule_freeze(epoch)

        for iteration, _ in enumerate(pbar):

            training_data = next(dataloader)
            training_data = move_data_to_device(training_data, self._state["devices"][0])

            self.optimzier.schedule_lr(epoch, iteration)
            self.optimizer.zero_grad()

            pred_data = self.model(training_data)

            loss_extra = [loss(pred_data, tarining_data)
                          for loss in self.losses]
            training_losses, extras = list(zip(*loss_extra))
            loss_weights = [loss.get_hps()["weight"] for loss in self.losses]

            total_loss = [loss * weight for loss, weight in zip(training_losses, loss_weights)]

            total_loss.backward()
            
            self.optimizer.step()

            # prompt
            # losses = []
            # for t in [cls_loss, ctr_loss, bbox_loss, iou]:
            #     losses.append(t)

            # print_str = 'epoch %d, ' % epoch
            # for l in losses:
            #     print_str += lib_repo.retrieve_name(l) + ': %.3f, ' % l.detach().cpu().numpy()
            # print_str += "lr: %.1e" % lr
            # pbar.set_description(print_str)

        # update state at the end of epoch
        self._state["epoch"] += 1

    def load_snapshot(self, snapshot_file):
        r""" 
        load snapshot
        """
        if osp.exists(snapshoto_file):
            # snapshot = torch.load(snapshoto_file, map_location=torch.device("cuda"))
            dev = self._state["devices"][0]
            snapshot = torch.load(snapshoto_file, map_location=dev)
            self.model.load_state_dict(snapshot["model_state_dict"])
            self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])
            self._state["epoch"] = snapshot["epoch"]+1
            logger.info("Load snapshot from:", osp.realpath(snapshoto_file))
        else:
            logger.info("%s does not exist, snapshot not loaded."%snapshot_file)

        # epoch_latest = './latest'
        # if osp.exists(epoch_latest):
        #     snapshot = torch.load(epoch_latest, map_location=torch.device("cuda"))
        #     model.load_state_dict(snapshot['model_state_dict'])
        #     optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        #     epoch = snapshot['epoch']+1
        #     print("Load snapshot from:", osp.realpath(epoch_latest))
        # else:
        #     epoch = 0

        print("Train from epoch %d" % self._state["epoch"])

    def save_snapshot(self,):
        r""" 
        save snapshot for current epoch
        """
        snapshot_dir = self._hyper_params["snapshot_dir"]
        epoch = self._state["epoch"]
        snapshot_file = osp.join(snapshot_dir, 
                                 "epoch-{}.pkl".format(epoch))
        snapshot_dict = {'epoch': epoch,
                         'model_state_dict': lib_repo.unwrap_model(model).state_dict(),
                         'optimizer_state_dict': self.optimizer.state_dict()}
        ensure_dir(snapshot_dir)
        torch.save(snapshot_dict, snapshot_file)
        while not osp.exists(snapshot_file):
            logger.info("retrying")
            torch.save(snapshot_dict, snapshot_file)
        logger.info("Snapshot saved at:", snapshot_file)
