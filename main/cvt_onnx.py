# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip

import argparse
import os.path as osp

from loguru import logger

import onnx
import onnxruntime
import torch
import torch.onnx

import numpy as np

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.model import builder as model_builder
from videoanalyst.utils import complete_path_wt_root_in_cfg


def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg',
                        '--config',
                        default='',
                        type=str,
                        help='experiment configuration')
    parser.add_argument('-o',
                        '--output',
                        default='',
                        type=str,
                        help='output onnx file name')

    return parser


def to_numpy(tensor):
    return tensor.detach().cpu().numpy(
    ) if tensor.requires_grad else tensor.cpu().numpy()


def export_siamfcpp_fea_onnx(task_cfg, parsed_args):
    # build model
    model = model_builder.build("track", task_cfg.model)
    model.eval()
    # extract feature extractor onnx model
    batch_size = 1
    x = torch.randn(batch_size, 3, 127, 127)
    fea = model(x, phase="feature")
    output_path = parsed_args.output + "_fea.onnx"
    torch.onnx.export(
        model,
        x,
        output_path,
        verbose=False,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["target_img"],
        output_names=["c_z_k", "r_z_k"],
        dynamic_axes={
            'target_img': {
                0: 'batch_size'
            },  # variable lenght axes
            'c_z_k': {
                0: 'batch_size'
            },
            'r_z_k': {
                0: 'batch_size'
            }
        })
    # check onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(output_path)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(fea[0]),
                               ort_outs[0],
                               rtol=1e-03,
                               atol=1e-05)


def export_siamfcpp_track_onnx(task_cfg, parsed_args):
    # build model
    model = model_builder.build("track", task_cfg.model)
    model.eval()
    # extract feature extractor onnx model
    batch_size = 1
    search_im = torch.randn(batch_size, 3, 303, 303)
    c_z_k = torch.randn(batch_size, 256, 4, 4)
    r_z_k = torch.randn(batch_size, 256, 4, 4)
    fea = model(search_im, c_z_k, r_z_k, phase="onnx_track")
    output_path = parsed_args.output + "_track.onnx"
    torch.onnx.export(model, (search_im, c_z_k, r_z_k),
                      output_path,
                      verbose=False,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=["search_img", "c_z_k", "r_z_k"],
                      output_names=["cls", "ctr", "box"])
    # check onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(output_path)
    ort_inputs = {
        ort_session.get_inputs()[0].name: to_numpy(search_im),
        ort_session.get_inputs()[1].name: to_numpy(c_z_k),
        ort_session.get_inputs()[2].name: to_numpy(r_z_k)
    }
    ort_outs = ort_session.run(None, ort_inputs)
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(fea[0]),
                               ort_outs[0],
                               rtol=1e-03,
                               atol=1e-05)
    np.testing.assert_allclose(to_numpy(fea[1]),
                               ort_outs[1],
                               rtol=1e-03,
                               atol=1e-05)


if __name__ == '__main__':
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()

    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)

    # resolve config
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()
    export_siamfcpp_track_onnx(task_cfg, parsed_args)
