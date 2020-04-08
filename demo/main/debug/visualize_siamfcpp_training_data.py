# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip

import argparse
from loguru import logger
import os.path as osp
import pickle
import numpy as np
import cv2

import torch

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.data import builder as dataloader_builder
from videoanalyst.data.dataset import builder as dataset_buidler
from videoanalyst.data.datapipeline import builder as datapipeline_builder
from videoanalyst.engine import builder as engine_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.model.loss import builder as losses_builder
from videoanalyst.optim import builder as optim_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.utils import Timer, ensure_dir, complete_path_wt_root_in_cfg, convert_tensor_to_numpy

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
        '-cfg', '--config',
        default='experiments/siamfcpp/data/siamfcpp_data-trn.yaml',
        type=str,
        help='path to experiment configuration')

    parser.add_argument(
        '-t', '--target',
        default='dataloader',
        type=str,
        help='targeted debugging module (dataloder|datasampler|dataset))')

    parser.add_argument(
        '-dt',
        '--data_type',
        default='bbox',
        type=str,
        help='target datat yep(bbox | mask))')

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
    #task_cfg.data.sampler.submodules.dataset.GOT10kDataset.check_integrity = False
    task_cfg.freeze()

    if parsed_args.target == "dataloader":
        logger.info("visualize for dataloader")
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
                target_cfg = task_cfg.data.target
                show_img_FCOS(target_cfg[target_cfg.name], training_sample)
    elif parsed_args.target == "dataset":
        logger.info("visualize for dataset")
        from videoanalyst.utils import load_image
        import numpy as np
        datasets = dataset_buidler.build(
            task, task_cfg.data.sampler.submodules.dataset)
        dataset = datasets[0]
        while True:
            # pick a frame randomly
            seq_idx = np.random.choice(range(len(dataset)))
            seq_idx = int(seq_idx)
            seq = dataset[seq_idx]
            # for one video
            if len(seq['image']) > 1: 
                frame_idx = np.random.choice(range(len(seq['image'])))
                frame = {k: seq[k][frame_idx] for k in seq}
            else:
                frame = dict(image= seq['image'][0],anno=seq['anno'])
            # fetch & visualize data
            im = load_image(frame['image'])
            if parsed_args.data_type == "bbox":
                cv2.rectangle(im,
                            tuple(map(int, frame['anno'][:2])),
                            tuple(map(int, frame['anno'][2:])), (0, 255, 0),
                            thickness=3)
                im = cv2.resize(im, (0, 0), fx=0.33, fy=0.33)
                cv2.imshow("im", im)
                cv2.waitKey(0)
            elif parsed_args.data_type == "mask":
                cv2.imshow("im", im)
                print(frame['anno'][0])
                mask = (frame['anno'][0]*50).astype(np.uint8).copy()
                cv2.imwrite("mask_0.png", mask)
                cv2.waitKey(0)
            else:
                logger.error("data type {} is not support now".format(parsed_args.data_type))
                exit()

    elif parsed_args.target == "datapipeline":
        logger.info("visualize for datapipeline")
        datapipeline = datapipeline_builder.build(task, task_cfg.data, seed=1)
        target_cfg = task_cfg.data.target
        for i in range(5):
            sampled_data = datapipeline[0]
            print(sampled_data)
            #print(sampled_data.keys())
            #sampled_data['data1'] = convert_tensor_to_numpy(sampled_data['data1'])
            #sampled_data['data2'] = convert_tensor_to_numpy(sampled_data['data2'])
            #cv2.imwrite('data1.png', sampled_data['data1']['image'])
            #cv2.imwrite('data1mask.png', sampled_data['data1']['anno']*250)
            #cv2.imwrite('data2.png', sampled_data['data2']['image'])
            #cv2.imwrite('data2mask.png', sampled_data['data2']['anno']*250)
            data = convert_tensor_to_numpy(sampled_data)
            cv2.imwrite("z.png", data["im_z"].astype(np.uint8))
            cv2.imwrite("x.png", data["im_x"].astype(np.uint8))
            cv2.imwrite("g_img.png", data["global_img"].astype(np.uint8))
            cv2.imwrite("g_mask.png", data["global_mask"].astype(np.uint8)*250)
            cv2.imwrite("seg_img.png", data["seg_img"].astype(np.uint8))
            cv2.imwrite("filtered_g.png", data["filtered_global_img"].astype(np.uint8))
            cv2.imwrite("seg_mask.png", data["seg_mask"].astype(np.uint8)*250)
            #show_img_FCOS(
            #    target_cfg[target_cfg.name],
            #    sampled_data,
            #)
