# -*- coding: utf-8 -*

import numpy as np

# import lib_repo
import torch
import torch.nn as nn

from videoanalyst.pipeline.pipeline_base import PipelineBase
from videoanalyst.pipeline.tracker.tracker_base import TRACK_PIPELINES
from videoanalyst.pipeline.utils import (cxywh2xywh, get_crop,
                                         get_subwindow_tracking,
                                         imarray_to_tensor, tensor_to_numpy,
                                         xywh2cxywh, xyxy2cxywh)

eps = 1e-7

# ============================== Tracker config for evaluation ============================== #
# tracker_config = lib_repo.Config()
# C = tracker_config
# C.total_stride = 8
# C.context_amount = 0.5
# C.penalty_k = 0.04
# C.window_influence = 0.21
# C.test_lr = 0.52
# C.windowing = 'cosine'
# C.z_size = 127
# C.x_size = 303
# C.head_shrink = 4
# C.score_size = (C.x_size - C.z_size) // C.total_stride + 1 - C.head_shrink
# C.score_offset = int((C.x_size - 1 - (C.score_size - 1) * C.total_stride) / 2)  # e.g. 43 for alexnet backbone


# ============================== Tracker definition ============================== #
@TRACK_PIPELINES.register
class SiamFCppTracker(PipelineBase):
    default_hyper_params = dict(
        total_stride=8,
        context_amount=0.5,
        penalty_k=0.04,
        window_influence=0.21,
        test_lr=0.52,
        windowing="cosine",
        z_size=127,
        x_size=303,
        head_shrink=6,
        min_w=10,
        min_h=10,
        phase_init="feature",
        phase_track="track",
    )

    #  calculated part
    # default_hyper_params['score_size'] = \
    #     (default_hyper_params['x_size']-['z_size']) // default_hyper_params['total_stride']\
    #     + 1 - default_hyper_params['head_shrink']
    # default_hyper_params['score_offset'] = \
    #     (default_hyper_params['x_size'] - 1 -
    #      (default_hyper_params['score_size'] - 1) * default_hyper_params['total_stride']) // 2

    def __init__(self, model=None, device=None, debug=False):
        super().__init__()
        self.update_params()

        # set underlying model to device
        if device is None:
            device = next(model.parameters()).device
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.debug = debug

    def update_params(self):
        hps = self._hyper_params
        hps['score_size'] = (hps['x_size'] - hps['z_size']
                             ) // hps['total_stride'] + 1 - hps['head_shrink']
        hps['score_offset'] = (
            hps['x_size'] - 1 -
            (hps['score_size'] - 1) * hps['total_stride']) // 2
        self._hyper_params = hps


    def feature(self, im, target_pos, target_sz, avg_chans=None):
        if avg_chans is None:
            avg_chans = np.mean(im, axis=(0, 1))

        z_size = self._hyper_params['z_size']
        context_amount = self._hyper_params['context_amount']

        im_z_crop, _ = get_crop(
            im,
            target_pos,
            target_sz,
            z_size,
            avg_chans=avg_chans,
            context_amount=context_amount,
            func_get_subwindow=get_subwindow_tracking,
        )
        phase = self._hyper_params['phase_init']
        with torch.no_grad():
            features = self.model(imarray_to_tensor(im_z_crop).to(self.device),
                                  phase=phase)

        return features, im_z_crop, avg_chans

    def init(self, im, state):
        """
        Initialize tracker
            Internal target state representation: self._state['state'] = (target_pos, target_sz)
        :param im: initial frame image
        :param state: bbox, format: xywh
        :return: None
        """
        rect = state  # bbox in xywh format is given for initialization in case of tracking
        box = xywh2cxywh(rect)
        target_pos, target_sz = box[:2], box[2:]

        self._state['im_h'] = im.shape[0]
        self._state['im_w'] = im.shape[1]

        # extract template feature
        features, im_z_crop, avg_chans = self.feature(im, target_pos, target_sz)

        score_size = self._hyper_params['score_size']
        if self._hyper_params['windowing'] == 'cosine':
            window = np.outer(np.hanning(score_size), np.hanning(score_size))
            window = window.reshape(-1)
        elif self._hyper_params['windowing'] == 'uniform':
            window = np.ones((score_size, score_size))
        else:
            window = np.ones((score_size, score_size))

        self._state['z_crop'] = im_z_crop
        self._state['avg_chans'] = avg_chans
        self._state['features'] = features
        self._state['window'] = window
        # self.state['target_pos'] = target_pos
        # self.state['target_sz'] = target_sz
        self._state['state'] = (target_pos, target_sz)

    def track(self,
              im_x,
              target_pos,
              target_sz,
              features,
              update_state=False,
              **kwargs):
        if 'avg_chans' in kwargs:
            avg_chans = kwargs['avg_chans']
        else:
            avg_chans = self._state['avg_chans']

        # x_pad = (self.config.x_size-self.config.z_size)//2
        z_size = self._hyper_params['z_size']
        x_size = self._hyper_params['x_size']
        context_amount = self._hyper_params['context_amount']
        phase_track = self._hyper_params['phase_track']
        im_x_crop, scale_x = get_crop(
            im_x,
            target_pos,
            target_sz,
            z_size,
            x_size=x_size,
            avg_chans=avg_chans,
            context_amount=context_amount,
            func_get_subwindow=get_subwindow_tracking,
        )
        with torch.no_grad():
            score, box, cls, ctr, *args = self.model(
                imarray_to_tensor(im_x_crop).to(self.device),
                *features,
                phase=phase_track)

        box = tensor_to_numpy(box[0])
        score = tensor_to_numpy(score[0])[:, 0]
        cls = tensor_to_numpy(cls[0])
        ctr = tensor_to_numpy(ctr[0])
        box_wh = xyxy2cxywh(box)

        # score post-processing
        best_pscore_id, pscore, penalty = self._postprocess_score(
            score, box_wh, target_sz, scale_x)
        # box post-processing
        new_target_pos, new_target_sz = self._postprocess_box(
            best_pscore_id, score, box_wh, target_pos, target_sz, scale_x,
            x_size, penalty)

        if self.debug:
            box = self._cvt_box_crop2frame(box_wh, target_pos, x_size, scale_x)

        # restrict new_target_pos & new_target_sz
        new_target_pos, new_target_sz = self._restrict_box(
            new_target_pos, new_target_sz)

        # record basic mid-level info
        self._state['x_crop'] = im_x_crop
        bbox_pred_in_crop = np.rint(box[best_pscore_id]).astype(np.int)
        self._state['bbox_pred_in_crop'] = bbox_pred_in_crop
        # record optional mid-level info
        if update_state:
            self._state['score'] = score
            self._state['pscore'] = pscore
            self._state['all_box'] = box
            self._state['cls'] = cls
            self._state['ctr'] = ctr

        return new_target_pos, new_target_sz

    def update(self, im):

        # get track
        # target_pos_prior, target_sz_prior = self.state['target_pos'], self.state['target_sz']
        target_pos_prior, target_sz_prior = self._state['state']
        features = self._state['features']

        # forward inference to estimate new state
        target_pos, target_sz = self.track(im,
                                           target_pos_prior,
                                           target_sz_prior,
                                           features,
                                           update_state=True)

        # save underlying state
        # self.state['target_pos'], self.state['target_sz'] = target_pos, target_sz
        self._state['state'] = target_pos, target_sz

        # return rect format
        track_rect = cxywh2xywh(np.concatenate([target_pos, target_sz],
                                               axis=-1))
        return track_rect

    # ======== tracking processes ======== #

    def _postprocess_score(self, score, box_wh, target_sz, scale_x):
        """
        Perform SiameseRPN-based tracker's post-processing of score
        :param score: (HW, ), score prediction
        :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
        :param target_sz: previous state (w & h)
        :param scale_x:
        :return:
            best_pscore_id: index of chosen candidate along axis HW
            pscore: (HW, ), penalized score
            penalty: (HW, ), penalty due to scale/ratio change
        """
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # size penalty
        penalty_k = self._hyper_params['penalty_k']
        target_sz_in_crop = target_sz * scale_x
        s_c = change(
            sz(box_wh[:, 2], box_wh[:, 3]) /
            (sz_wh(target_sz_in_crop)))  # scale penalty
        r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) /
                     (box_wh[:, 2] / box_wh[:, 3]))  # ratio penalty
        penalty = np.exp(-(r_c * s_c - 1) * penalty_k)
        pscore = penalty * score

        # ipdb.set_trace()
        # cos window (motion model)
        window_influence = self._hyper_params['window_influence']
        pscore = pscore * (
            1 - window_influence) + self._state['window'] * window_influence
        best_pscore_id = np.argmax(pscore)

        return best_pscore_id, pscore, penalty

    def _postprocess_box(self, best_pscore_id, score, box_wh, target_pos,
                         target_sz, scale_x, x_size, penalty):
        """
        Perform SiameseRPN-based tracker's post-processing of box
        :param score: (HW, ), score prediction
        :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
        :param target_pos: (2, ) previous position (x & y)
        :param target_sz: (2, ) previous state (w & h)
        :param scale_x: scale of cropped patch of current frame
        :param x_size: size of cropped patch
        :param penalty: scale/ratio change penalty calculated during score post-processing
        :return:
            new_target_pos: (2, ), new target position
            new_target_sz: (2, ), new target size
        """
        pred_in_crop = box_wh[best_pscore_id, :] / np.float32(scale_x)
        # about np.float32(scale_x)
        # attention!, this casting is done implicitly
        # which can influence final EAO heavily given a model & a set of hyper-parameters

        # box post-postprocessing
        test_lr = self._hyper_params['test_lr']
        lr = penalty[best_pscore_id] * score[best_pscore_id] * test_lr
        res_x = pred_in_crop[0] + target_pos[0] - (x_size // 2) / scale_x
        res_y = pred_in_crop[1] + target_pos[1] - (x_size // 2) / scale_x
        res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
        res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

        new_target_pos = np.array([res_x, res_y])
        new_target_sz = np.array([res_w, res_h])

        return new_target_pos, new_target_sz

    def _restrict_box(self, target_pos, target_sz):
        """
        Restrict target position & size
        :param target_pos: (2, ), target position
        :param target_sz: (2, ), target size
        :return:
            target_pos, target_sz
        """
        target_pos[0] = max(0, min(self._state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(self._state['im_h'], target_pos[1]))
        target_sz[0] = max(self._hyper_params['min_w'],
                           min(self._state['im_w'], target_sz[0]))
        target_sz[1] = max(self._hyper_params['min_h'],
                           min(self._state['im_h'], target_sz[1]))

        return target_pos, target_sz

    def _cvt_box_crop2frame(self, box_in_crop, target_pos, scale_x, x_size):
        """
        Convert box from cropped patch to original frame
        :param box_in_crop: (4, ), cxywh, box in cropped patch
        :param target_pos: target position
        :param scale_x: scale of cropped patch
        :param x_size: size of cropped patch
        :return:
            box_in_frame: (4, ), cxywh, box in original frame
        """
        x = (box_in_crop[...,
                         0]) / scale_x + target_pos[0] - (x_size // 2) / scale_x
        y = (box_in_crop[...,
                         1]) / scale_x + target_pos[1] - (x_size // 2) / scale_x
        w = box_in_crop[..., 2] / scale_x
        h = box_in_crop[..., 3] / scale_x
        box_in_frame = np.stack([x, y, w, h], axis=-1)

        return box_in_frame
