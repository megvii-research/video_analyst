# -*- coding: utf-8 -*
import copy
import itertools
import logging
import math
import os
import os.path as osp
import time

from collections import OrderedDict
from os.path import join

from typing import List

import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..trainer_base import TRACK_TRAINERS, TrainerBase

from videoanalyst.model.module_base import ModuleBase
from videoanalyst.optim.optimizer.optimizer_base import OptimizerBase
from videoanalyst.utils import ensure_dir, Timer, move_data_to_device, unwrap_model

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
    extra_hyper_params = dict(
        devices=["cpu"],
        num_iterations=1,
        max_epoch=1,
        snapshot="",
    )

    def __init__(self, optimizer, dataloader, monitors=[]):
        r"""
        Crete tester with config and pipeline

        Arguments
        ---------
        optimizer: ModuleBase
            including optimizer, model and loss
        dataloder: DataLoader
            PyTorch dataloader object. 
            Usage: batch_data = next(dataloader)
        """
        super(RegularTrainer, self).__init__(optimizer, dataloader, monitors)
        # update state
        self._state["epoch"] = -1  # uninitialized

    def update_params(self, ):
        super(RegularTrainer, self).update_params()
        self._state["devices"] = [
            torch.device(dev) for dev in self._hyper_params["devices"]
        ]
        self._state["snapshot_dir"] = osp.join(self._hyper_params["exp_save"],
                                               self._hyper_params["exp_name"])
        self.init_train()

    def init_train(self, ):
        torch.cuda.empty_cache()

        devs = self._state["devices"]
        self._model.train()
        self._model.to(devs[0])

        for k in self._losses:
            self._losses[k].to(devs[0])

        self.load_snapshot()

        if len(self._state["devices"]) > 1:
            self._model = nn.DataParallel(self._model, device_ids=devs)
            logger.info("Use nn.DataParallel for data parallelism")

        super(RegularTrainer, self).init_train()

    def train(self):
        # epoch counter +1
        self._state["epoch"] += 1
        epoch = self._state["epoch"]
        max_epoch = self._hyper_params["max_epoch"]
        num_iterations = self._hyper_params["num_iterations"]

        self._optimizer.modify_grad(epoch)
        pbar = tqdm(range(num_iterations))
        self._state["pbar"] = pbar
        self._state["print_str"] = ""

        time_dict = OrderedDict()
        for iteration, _ in enumerate(pbar):
            with Timer(name="data", output_dict=time_dict):
                training_data = next(self._dataloader)

            training_data = move_data_to_device(training_data,
                                                self._state["devices"][0])

            schedule_info = self._optimizer.schedule(epoch, iteration)
            self._optimizer.zero_grad()

            # forward propagation
            with Timer(name="fwd", output_dict=time_dict):
                pred_data = self._model(training_data)

            # compute losses
            loss_extra_dict = OrderedDict()
            for k in self._losses:
                loss_extra_dict[k] = self._losses[k](pred_data, training_data)

            # split losses & extras
            training_losses, extras = OrderedDict(), OrderedDict()
            for k in self._losses:
                training_losses[k], extras[k] = loss_extra_dict[k]

            # get loss weights & sum up
            loss_weights = OrderedDict()
            for k in self._losses:
                loss_weights[k] = self._losses[k].get_hps()["weight"]
            total_loss = [
                training_losses[k] * loss_weights[k] for k in self._losses
            ]
            total_loss = sum(total_loss)

            # backward propagation
            with Timer(name="bwd", output_dict=time_dict):
                total_loss.backward()
            self._optimizer.modify_grad(epoch, iteration)
            with Timer(name="optim", output_dict=time_dict):
                self._optimizer.step()

            trainer_data = dict(
                schedule_info=schedule_info,
                training_losses=training_losses,
                extras=extras,
                time_dict=time_dict,
            )

            for monitor in self._monitors:
                monitor.update(trainer_data)

            print_str = self._state["print_str"]
            pbar.set_description(print_str)

    def is_completed(self):
        is_completed = (self._state["epoch"] + 1 >=
                        self._hyper_params["max_epoch"])
        return is_completed

    def load_snapshot(self, snapshot_file=""):
        r""" 
        load snapshot
        """
        if len(snapshot_file) <= 0:
            snapshot_file = self._hyper_params["snapshot"]
        if osp.exists(snapshot_file):
            # snapshot = torch.load(snapshoto_file, map_location=torch.device("cuda"))
            dev = self._state["devices"][0]
            snapshot = torch.load(snapshot_file, map_location=dev)
            self._model.load_state_dict(snapshot["model_state_dict"])
            self._optimizer.load_state_dict(snapshot["optimizer_state_dict"])
            self._state["epoch"] = snapshot["epoch"] + 1
            logger.info("Load snapshot from: %s" % osp.realpath(snapshot_file))
        else:
            logger.info("%s does not exist, no snapshot loaded." %
                        snapshot_file)

        logger.info("Train from epoch %d" % (self._state["epoch"] + 1))

    def save_snapshot(self, ):
        r""" 
        save snapshot for current epoch
        """
        snapshot_dir = self._state["snapshot_dir"]
        epoch = self._state["epoch"]
        snapshot_file = osp.join(snapshot_dir, "epoch-{}.pkl".format(epoch))
        snapshot_dict = {
            'epoch': epoch,
            'model_state_dict': unwrap_model(self._model).state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict()
        }
        ensure_dir(snapshot_dir)
        torch.save(snapshot_dict, snapshot_file)
        while not osp.exists(snapshot_file):
            logger.info("retrying")
            torch.save(snapshot_dict, snapshot_file)
        logger.info("Snapshot saved at: %s" % snapshot_file)


RegularTrainer.default_hyper_params = copy.deepcopy(
    RegularTrainer.default_hyper_params)
RegularTrainer.default_hyper_params.update(RegularTrainer.extra_hyper_params)
