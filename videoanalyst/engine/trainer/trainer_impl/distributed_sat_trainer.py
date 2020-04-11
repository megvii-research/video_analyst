# -*- coding: utf-8 -*
from typing import Tuple, List
import copy
import itertools
import time
from loguru import logger
import os.path as osp
from collections import OrderedDict

import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.distributed as dist

from videoanalyst.model.module_base import ModuleBase
from videoanalyst.optim.optimizer.optimizer_base import OptimizerBase
from videoanalyst.utils import (Timer, ensure_dir, move_data_to_device,
                                unwrap_model, average_gradients)
from videoanalyst.utils import dist_utils

from ..trainer_base import VOS_TRAINERS, TrainerBase


@VOS_TRAINERS.register
class DistributedSATTrainer(TrainerBase):
    r"""
    Distributed Trainer to test the vot dataset, the result is saved as follows
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
        minibatch=1,
        nr_image_per_epoch=1,
        max_epoch=1,
        snapshot="",
    )

    def __init__(self, optimizer, dataloader, monitors=[], tracker=None):
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
        super(DistributedSATTrainer, self).__init__(optimizer, dataloader,
                                                        monitors)
        # update state
        self._state["epoch"] = -1  # uninitialized
        self._state["initialized"] = False
        self._state["devices"] = torch.device("cuda:0")
        self.tracker = tracker

    def update_params(self, ):
        super(DistributedSATTrainer, self).update_params()
        self._hyper_params["num_iterations"] = self._hyper_params[
            "nr_image_per_epoch"] // self._hyper_params["minibatch"]
        self._state["snapshot_dir"] = osp.join(self._hyper_params["exp_save"],
                                               self._hyper_params["exp_name"])

        self._state["snapshot_file"] = self._hyper_params["snapshot"]

    def init_train(self, ):
        torch.cuda.empty_cache()
        devs = self._state["devices"]
        self._model.train()
        # load from self._state["snapshot_file"]
        self.load_snapshot()
        # parallelism with Distributed Data Parallel (DDP)
        self._model.set_device(devs[0])
        self._model = nn.parallel.DistributedDataParallel(
            self._model, device_ids=devs, find_unused_parameters=True
        )  # TODO: devs should be calculated based on rank & num_workers
        self.tracker.eval()
        self.tracker.set_device(devs[0])
        logger.info("Use nn.parallel.DistributedDataParallel for parallelism")
        super(DistributedSATTrainer, self).init_train()
        logger.info("{} initialized".format(type(self).__name__))

    def train(self):
        if not self._state["initialized"]:
            self.init_train()
        self._state["initialized"] = True

        # epoch counter +1
        self._state["epoch"] += 1
        epoch = self._state["epoch"]
        num_iterations = self._hyper_params["num_iterations"]

        # udpate engine_state
        self._state["max_epoch"] = self._hyper_params["max_epoch"]
        self._state["max_iteration"] = num_iterations

        self._optimizer.modify_grad(epoch)
        # TODO: build stats gathering code and reorganize tqdm
        self._state["print_str"] = ""

        time_dict = OrderedDict()
        for iteration in range(num_iterations):
            start_time = time.time()
            self._state["iteration"] = iteration
            with Timer(name="data", output_dict=time_dict):
                training_data = next(self._dataloader)
            training_data = move_data_to_device(training_data,
                                                self._state["devices"][0])
            schedule_info = self._optimizer.schedule(epoch, iteration)
            self._optimizer.zero_grad()
            with Timer(name="track_fwd", output_dict=time_dict):
                with torch.no_grad():
                    tracker_output = self.tracker(training_data, phase="train")
                corr_fea = tracker_output["corr_fea"].detach()
            # forward propagation
            with Timer(name="segfwd", output_dict=time_dict):
                predict_data = self._model(training_data["seg_img"], corr_fea, training_data["filtered_global_img"])
                training_losses, extras = OrderedDict(), OrderedDict()
                for loss_name, loss in self._losses.items():
                    training_losses[loss_name], extras[loss_name] = loss(
                        predict_data, training_data["seg_mask"])
                total_loss = sum(training_losses.values())
            # backward propagation
            with Timer(name="bwd", output_dict=time_dict):
                total_loss.backward()
            # TODO: No need for average_gradients() when wrapped model with DDP?
            # TODO: need to register _optimizer.modify_grad as hook
            #       see https://discuss.pytorch.org/t/distributeddataparallel-modify-gradient-before-averaging/59291
            # self._optimizer.modify_grad(epoch, iteration)
            with Timer(name="optim", output_dict=time_dict):
                self._optimizer.step()
            cost_time = (num_iterations-iteration)*(time.time() - start_time)
            if dist_utils.get_rank() == 0:
                trainer_data = dict(
                    schedule_info=schedule_info,
                    training_losses=training_losses,
                    training_data=training_data,
                    extras=extras,
                    time_dict=time_dict,
                    predict_data=predict_data,
                    iter=iteration,
                )
                for monitor in self._monitors:
                    monitor.update(trainer_data)
                print_str = "{}/{} epoch {} eta ({}h {}m {}s) bs: {} ".format(iteration, num_iterations, epoch, int(cost_time//(3600)), int(cost_time%3600//60), int(cost_time%60), training_data["im_x"].size(0))+self._state["print_str"]
                logger.info(print_str)
            del training_data 

DistributedSATTrainer.default_hyper_params = copy.deepcopy(
    DistributedSATTrainer.default_hyper_params)
DistributedSATTrainer.default_hyper_params.update(
    DistributedSATTrainer.extra_hyper_params)
