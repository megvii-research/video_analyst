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
from torch import nn
from torch.utils.data import DataLoader
torch.backends.cudnn.enabled = False

from ..trainer_base import TRACK_TRAINERS, TrainerBase
from videoanalyst.utils import ensure_dir, move_data_to_device, unwrap_model

from videoanalyst.model.module_base import ModuleBase
from videoanalyst.optim.optimizer.optimizer_base import OptimizerBase

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

    def __init__(self, ):
        r"""
        Crete tester with config and pipeline

        # Arguments
        # ---------
        # model: ModuleBase
        # dataloder: DataLoader
        # losses: ModuleBase
        # processes: ProcessBase

        """
        super(RegularTrainer, self).__init__()
        # famous four elements in Deep Laerning (c.f. <Deep Learning>, Goodfellow et al.)
        # update state
        self._state["epoch"] = -1  # uninitialized
        self.update_params()

    def update_params(self, ):
        self._state["devices"] = [torch.device(dev) for dev in self._hyper_params["devices"]]
        self._state["snapshot_dir"] = osp.join(self._hyper_params["exp_save"],
                                               self._hyper_params["exp_name"])

    def init_train(self, ):
        devs = self._state["devices"]

        self._optimizer.set_model(self._model)

        self._model.train()
        self._model.to(devs[0])
        # torch.cuda.empty_cache()

        if len(self._state["devices"]) > 1:
            self._model = nn.DataParallel(self._model, device_ids=devs)
            logger.info("Use nn.DataParallel for data parallelism")

        for k in self._losses:
            self._losses[k].to(devs[0])

        self._optimizer.build_optimizer()

    def train(self):
        # epoch counter +1
        self._state["epoch"] += 1
        epoch = self._state["epoch"]
        max_epoch = self._hyper_params["max_epoch"]
        num_iterations = self._hyper_params["num_iterations"]

        # self._optimizer.schedule_freeze(epoch)
        self._optimizer.modify_grad(epoch)

        pbar = tqdm(range(num_iterations))

        for iteration, _ in enumerate(pbar):

            training_data = next(self._dataloader)
            training_data = move_data_to_device(training_data, self._state["devices"][0])

            schedule_info = self._optimizer.schedule(epoch, iteration)
            # from IPython import embed;embed()
            self._optimizer.zero_grad()

            # forward propagation
            pred_data = self._model(training_data)

            loss_extra_dict = OrderedDict()
            for k in self._losses:
                loss_extra_dict[k] = self._losses[k](pred_data, training_data)

            training_losses, extras = OrderedDict(), OrderedDict()
            for k in self._losses:
                training_losses[k], extras[k] = loss_extra_dict[k]

            loss_weights = OrderedDict()
            for k in self._losses:
                loss_weights[k] = self._losses[k].get_hps()["weight"]

            total_loss = [training_losses[k] * loss_weights[k] for k in self._losses]
            total_loss = sum(total_loss)

            # from IPython import embed;embed()
            # backward propagation
            total_loss.backward()
            self._optimizer.modify_grad(epoch, iteration)
            
            self._optimizer.step()

            # prompt
            print_str = 'epoch %d, ' % epoch
            for k in schedule_info:
                print_str +=  '%s: %.1e, ' % (k, schedule_info[k])
            for k in training_losses:
                l = training_losses[k]
                print_str +=  '%s: %.3f, ' % (k, l.detach().cpu().numpy())
            # print_str += "lr: %.1e" % lr
            pbar.set_description(print_str)


    def is_completed(self):
        is_completed = (self._state["epoch"]+1 >= self._hyper_params["max_epoch"])
        return is_completed

    def load_snapshot(self, snapshot_file):
        r""" 
        load snapshot
        """
        if osp.exists(snapshot_file):
            # snapshot = torch.load(snapshoto_file, map_location=torch.device("cuda"))
            dev = self._state["devices"][0]
            snapshot = torch.load(snapshot_file, map_location=dev)
            self._model.load_state_dict(snapshot["model_state_dict"])
            self._optimizer.load_state_dict(snapshot["optimizer_state_dict"])
            self._state["epoch"] = snapshot["epoch"]+1
            logger.info("Load snapshot from:", osp.realpath(snapshot_file))
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
        snapshot_dir = self._state["snapshot_dir"]
        epoch = self._state["epoch"]
        snapshot_file = osp.join(snapshot_dir, 
                                 "epoch-{}.pkl".format(epoch))
        snapshot_dict = {'epoch': epoch,
                         'model_state_dict': unwrap_model(self._model).state_dict(),
                         'optimizer_state_dict': self._optimizer.state_dict()}
        ensure_dir(snapshot_dir)
        torch.save(snapshot_dict, snapshot_file)
        while not osp.exists(snapshot_file):
            logger.info("retrying")
            torch.save(snapshot_dict, snapshot_file)
        logger.info("Snapshot saved at: %s" % snapshot_file)
