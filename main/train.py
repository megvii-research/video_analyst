# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip

import argparse
import logging
import os.path as osp

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task

from videoanalyst.engine import builder as engine_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.data import builder as dataloader_builder
from videoanalyst.model.loss import builder as losses_builder
from videoanalyst.optimizer import builder as optimizer_builder

from videoanalyst.pipeline import builder as pipeline_builder

logger = logging.getLogger('global')


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--config',
                        default='',
                        type=str,
                        help='experiment configuration')

    return parser

if __name__ == '__main__':
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()


    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)


    # resolve config
    root_cfg = root_cfg.train
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()

    # build model

    model = model_builder.build(task, task_cfg.model)
    dataloader = dataloader_builder.build(task, task_cfg.data)
    losses = losses_builder.build(task, task_cfg.model.losses)
    optimizer = optimizer_builder.build(task, task_cfg.optimizer, model)

    # build trainer
    trainer = engine_builder.build(task, task_cfg, "trainer")

    # bind model to optimizer
    optimizer.set_model(model)

    # set model in trainer
    trainer.set_model(model)
    trainer.set_dataloader(dataloader)
    trainer.set_losses(losses)
    trainer.set_optimizer(optimizer)


    trainer.init_train()
    # from IPython import embed;embed()
    while not trainer.is_completed():
        trainer.train()
        trainer.save_snapshot()

    logger.info("Training completed.")