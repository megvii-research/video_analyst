# -*- coding: utf-8 -*

from copy import deepcopy

import cv2
import numpy as np

import torch

from videoanalyst.pipeline.pipeline_base import TRACK_PIPELINES, PipelineBase
from videoanalyst.pipeline.utils import (cxywh2xywh, get_crop,
                                         get_subwindow_tracking,
                                         imarray_to_tensor, tensor_to_numpy,
                                         xywh2cxywh, xyxy2cxywh)

from ..utils.online_classifier.base_classifier import BaseClassifier


# ============================== Tracker definition ============================== #
@TRACK_PIPELINES.register
class SiamFCppOnlineTracker(PipelineBase):
    r"""
    Basic SiamFC++ tracker

    Hyper-parameters
    ----------------
        total_stride: int
            stride in backbone
        context_amount: float
            factor controlling the image patch cropping range. Set to 0.5 by convention.
        test_lr: float
            factor controlling target size updating speed
        penalty_k: float
            factor controlling the penalization on target size (scale/ratio) change
        window_influence: float
            factor controlling spatial windowing on scores
        windowing: str
            windowing type. Currently support: "cosine"
        z_size: int
            template image size
        x_size: int
            search image size
        num_conv3x3: int
            number of conv3x3 tiled in head
        min_w: float
            minimum width
        min_h: float
            minimum height
        phase_init: str
            phase name for template feature extraction
        phase_track: str
            phase name for target search
        corr_fea_output: bool
            whether output corr feature
        debug_show: bool
            whether show result in the tracking
        online_debug_show: bool
            debug for online module
        online_score_weight: float
            the online score weight 
        raw_fea_size: int
            the output size of the feature from backbone
        projection_reg: float
            Projection regularization factor
        use_projection_matrix: bool
            Use projection matrix, i.e. use the factorized convolution formulation
        update_projection_matrix: bool
            Whether the projection matrix should be optimized or not
        compressed_dim: int
            Dimension output of projection matrix
        proj_init_method: str
            Method for initializing the projection matrix
        projection_activation: str
            Activation function after projection ('none', 'relu', 'elu' or 'mlu')
        use_attention_layer: bool
            Whether use attention layer
        channel_attention: bool
            whether use channel-wise attention
        spatial_attention: str
            method of spatial-wise attention such ('none', 'pool')
        att_activation: str # none|relu|elu|mlu
            Activation function after attention ('none', 'relu', 'elu', 'mlu')
        filter_reg: float
            Filter regularization factor
        z_kernel_size: tuple
            Kernel size of filter
        filter_init_method: str
            Method for initializing the spatial filter
        reponse_activation: str or tuple
            Activation function on the output scores ('none', 'relu', 'elu' or 'mlu')
        use_augmentation: bool
            Whether use augmentation for examples for init training
        augmentation_expansion_factor: float
            How much to expand sample when doing augmentation
        augmentation_shift_factor: float
            How much random shift to do on each augmented sample
        augmentation_shift: bool
            whether use random shift in aug
        augmentation_scale: bool
            whether use random scale in aug
        augmentation_rotate: list
            the retate scales in aug
        augmentation_relativeshift: list
            the relative shift in aug
        augmentation_fliplr: bool
            whether use flip in aug
        augmentation_blur: list
            blur factor in aug
        augmentation_dropout: tuple
            (drop_img_num, drop_rate) in aug
        CG_optimizer: bool
            whether enable CG optimizer
        precond_learning_rate: float
            Learning rate
        init_samples_minimum_weight: float

        sample_memory_size: int
            Memory size
        output_sigma_factor: float
            Standard deviation of Gaussian label relative to target size
        # Gauss-Newton CG
        optimizer: str
            optimizer name
        init_CG_iter: int
            The total number of Conjugate Gradient iterations used in the first frame
        init_GN_iter: int
            The number of Gauss-Newton iterations used in the first frame (only if the projection matrix is updated)
        train_skipping: int
            How often to run training (every n-th frame)
        CG_iter: int
            The number of Conjugate Gradient iterations in each update after the first frame
        post_init_CG_iter: int
            CG iterations to run after GN
        fletcher_reeves: bool
            Use the Fletcher-Reeves (true) or Polak-Ribiere (false) formula in the Conjugate Gradient
        CG_forgetting_rate: bool
            Forgetting rate of the last conjugate direction
        #SGD
        optimizer_step_length: int
            Gradient step length in SGD
        optimizer_momentum: float
            Gradient momentum in SGD
        # advanced localization -hard negtive mining & absence assessment
        advanced_localization: bool
            Use this or not
        analyze_convergence: bool
        hard_negative_learning_rate: float
            Learning rate if hard negative detected
        hard_negative_CG_iter: int
            Number of optimization iterations to use if hard negative detected
        target_not_found_threshold: float
            Absolute score threshold to detect target missing
        target_neighborhood_scale: float
            Dispacement to consider for distractors
        distractor_threshold: float
            Relative threshold to find distractors
        displacement_scale: float
            Dispacement to consider for distractors
        hard_negative_threshold: float
            Relative threshold to find hard negative samples

    Hyper-parameters (to be calculated at runtime)
    ----------------------------------------------
    score_size: int
        final feature map
    score_offset: int
        final feature map
    """
    default_hyper_params = dict(
        # global set
        total_stride=8,
        score_size=17,
        score_offset=87,
        context_amount=0.5,
        test_lr=0.52,
        penalty_k=0.04,
        window_influence=0.21,
        windowing="cosine",
        z_size=127,
        x_size=303,
        num_conv3x3=3,
        min_w=10,
        min_h=10,
        phase_init="feature",
        phase_track="track",
        corr_fea_output=False,
        debug_show=False,
        online_debug_show=False,
        online_score_weight=0.5,
        # online model param
        projection_reg=1e-4,
        # first layer compression
        use_projection_matrix=True,
        update_projection_matrix=True,
        compressed_dim=128,
        proj_init_method="pca",
        projection_activation="none",  # relu|elu|none|mlu
        # second layer attention
        use_attention_layer=True,
        channel_attention=True,
        att_fc1_reg=1e-4,
        att_fc2_reg=1e-4,
        att_init_method="randn",
        spatial_attention="pool",
        att_activation="relu",  # none|relu|elu|mlu
        # third layer (filter)
        filter_reg=1e-1,
        raw_fea_size=26,
        z_kernel_size=(4, 4),
        filter_init_method="randn",  # zeros|"randn"
        reponse_activation=("mlu", 0.05),
        # augmentation params
        use_augmentation=True,
        augmentation_expansion_factor=2,
        augmentation_shift_factor=1 / 3,
        augmentation_shift=False,
        augmentation_scale=False,
        augmentation_rotate=[
            5, -5, 10, -10, 20, -20, 30, -30, 45, -45, -60, 60
        ],
        augmentation_relativeshift=[(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6),
                                    (-0.6, -0.6)],
        augmentation_fliplr=True,
        augmentation_blur=[(2, 0.2), (0.2, 2), (3, 1), (1, 3), (2, 2)],
        augmentation_dropout=(7, 0.2),
        # optimization params
        CG_optimizer=True,
        precond_learning_rate=0.01,
        init_samples_minimum_weight=0.25,
        sample_memory_size=250,
        output_sigma_factor=0.25,

        # Gauss-Newton CG
        optimizer='GaussNewtonCG',
        init_CG_iter=60,
        init_GN_iter=6,
        train_skipping=10,
        CG_iter=5,
        post_init_CG_iter=0,
        fletcher_reeves=False,
        CG_forgetting_rate=False,
        #SGD
        optimizer_step_length=10,
        optimizer_momentum=0.9,
        # advanced localization -hard negtive mining & absence assessment
        advanced_localization=True,
        analyze_convergence=False,
        hard_negative_learning_rate=0.02,
        hard_negative_CG_iter=5,
        target_not_found_threshold=0.25,
        target_neighborhood_scale=2.2,
        distractor_threshold=0.8,
        displacement_scale=0.8,
        hard_negative_threshold=0.5,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_params()

        # set underlying model to device
        self.device = torch.device("cpu")
        self.debug = False
        self.set_model(self._model)
        self.lost_count = 0

    def set_model(self, model):
        """model to be set to pipeline. change device & turn it into eval mode
        
        Parameters
        ----------
        model : ModuleBase
            model to be set to pipeline
        """
        self._model = model.to(self.device)
        self._model.eval()

    def set_device(self, device):
        self.device = device
        self._model = self._model.to(device)
        self.online_classifier.device = device

    def update_params(self):
        hps = self._hyper_params
        hps['score_size'] = (
            hps['x_size'] -
            hps['z_size']) // hps['total_stride'] + 1 - hps['num_conv3x3'] * 2
        hps['score_offset'] = (
            hps['x_size'] - 1 -
            (hps['score_size'] - 1) * hps['total_stride']) // 2
        self._hyper_params = hps
        self.online_classifier = BaseClassifier(self._model, hps)

    def feature(self, im: np.array, target_pos, target_sz, avg_chans=None):
        """Extract feature

        Parameters
        ----------
        im : np.array
            initial frame
        target_pos : 
            target position (x, y)
        target_sz : [type]
            target size (w, h)
        avg_chans : [type], optional
            channel mean values, (B, G, R), by default None
        
        Returns
        -------
        [type]
            [description]
        """
        if avg_chans is None:
            avg_chans = np.mean(im, axis=(0, 1))

        z_size = self._hyper_params['z_size']
        context_amount = self._hyper_params['context_amount']

        im_z_crop, scale_z = get_crop(
            im,
            target_pos,
            target_sz,
            z_size,
            avg_chans=avg_chans,
            context_amount=context_amount,
            func_get_subwindow=get_subwindow_tracking,
        )
        self.scale_z = scale_z
        phase = self._hyper_params['phase_init']
        with torch.no_grad():
            data = imarray_to_tensor(im_z_crop).to(self.device)
            features = self._model(data, phase=phase)

        return features, im_z_crop, avg_chans, data

    def init(self, im, state):
        r"""Initialize tracker
            Internal target state representation: self._state['state'] = (target_pos, target_sz)
        
        Arguments
        ---------
        im : np.array
            initial frame image
        state
            target state on initial frame (bbox in case of SOT), format: xywh
        """
        self.frame_num = 1
        self.temp_max = 0
        rect = state  # bbox in xywh format is given for initialization in case of tracking
        box = xywh2cxywh(rect)
        target_pos, target_sz = box[:2], box[2:]

        self._state['im_h'] = im.shape[0]
        self._state['im_w'] = im.shape[1]

        # extract template feature
        features, im_z_crop, avg_chans, im_z_crop_t = self.feature(
            im, target_pos, target_sz)

        score_size = self._hyper_params['score_size']
        if self._hyper_params['windowing'] == 'cosine':
            window = np.outer(np.hanning(score_size), np.hanning(score_size))
            window = window.reshape(-1)
        elif self._hyper_params['windowing'] == 'uniform':
            window = np.ones((score_size, score_size))
        else:
            window = np.ones((score_size, score_size))

        self._state['z_crop'] = im_z_crop
        self._state['z0_crop'] = im_z_crop_t
        with torch.no_grad():
            self._model.instance(im_z_crop_t)
        self._state['avg_chans'] = avg_chans
        self._state['features'] = features
        self._state['window'] = window
        self._state['state'] = (target_pos, target_sz)
        # init online classifier
        z_size = self._hyper_params['z_size']
        x_size = self._hyper_params['x_size']
        context_amount = self._hyper_params['context_amount']
        init_im_crop, scale_x = get_crop(
            im,
            target_pos,
            target_sz,
            z_size,
            x_size=x_size * 2,
            avg_chans=avg_chans,
            context_amount=context_amount,
            func_get_subwindow=get_subwindow_tracking,
        )
        init_x_crop_t = imarray_to_tensor(init_im_crop)
        self.online_classifier.initialize(init_x_crop_t, state)

    def get_avg_chans(self):
        return self._state['avg_chans']

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
        self._state["scale_x"] = deepcopy(scale_x)
        with torch.no_grad():
            score, box, cls, ctr, extra = self._model(
                imarray_to_tensor(im_x_crop).to(self.device),
                *features,
                phase=phase_track)
        if self._hyper_params["corr_fea_output"]:
            self._state["corr_fea"] = extra["corr_fea"]

        box = tensor_to_numpy(box[0])
        score = tensor_to_numpy(score[0])[:, 0]
        cls = tensor_to_numpy(cls[0])
        ctr = tensor_to_numpy(ctr[0])

        def normalize(score):
            score = (score - np.min(score)) / (np.max(score) - np.min(score))
            return score

        if True:
            flag, s = self.online_classifier.track()
            if flag == 'not_found':
                self.lost_count += 1
            else:
                self.lost_count = 0

            confidence = s.detach().cpu().numpy()
            offset = (confidence.shape[0] -
                      self._hyper_params["score_size"]) // 2
            confidence = confidence[offset:-offset, offset:-offset]
            confidence = normalize(confidence).flatten()
        box_wh = xyxy2cxywh(box)

        # score post-processing
        best_pscore_id, pscore, penalty = self._postprocess_score(
            score, confidence, box_wh, target_sz, scale_x)
        if self._hyper_params["debug_show"]:
            bbox_in_crop = box[best_pscore_id, :]
            bbox_in_crop = tuple(map(int, bbox_in_crop))
            show_im_patch = im_x_crop.copy()
            cv2.rectangle(show_im_patch, bbox_in_crop[:2], bbox_in_crop[2:],
                          (0, 255, 0), 2)
            cv2.imshow("pred in crop", show_im_patch)
            # offline score
            score_mat = score.reshape(self._hyper_params["score_size"],
                                      self._hyper_params["score_size"])
            score_mat = (255 * score_mat).astype(np.uint8)
            score_map = cv2.applyColorMap(score_mat, cv2.COLORMAP_JET)
            cv2.imshow("offline score", score_map)
            score_mat = confidence.reshape(self._hyper_params["score_size"],
                                           self._hyper_params["score_size"])
            score_mat = (255 * score_mat).astype(np.uint8)
            score_map = cv2.applyColorMap(score_mat, cv2.COLORMAP_JET)
            cv2.imshow("online score", score_map)
            cv2.waitKey()

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
        self.online_classifier.update(
            np.concatenate([new_target_pos, new_target_sz], axis=0),
            self.scale_z, flag)
        # record optional mid-level info
        if update_state:
            self._state['score'] = score
            self._state['pscore'] = pscore[best_pscore_id]
            self._state['all_box'] = box
            self._state['cls'] = cls
            self._state['ctr'] = ctr

        return new_target_pos, new_target_sz

    def set_state(self, state):
        self._state["state"] = state

    def get_track_score(self):
        return float(self._state["pscore"])

    def update(self, im, state=None):
        """ Perform tracking on current frame
            Accept provided target state prior on current frame
            e.g. search the target in another video sequence simutanously

        Arguments
        ---------
        im : np.array
            current frame image
        state
            provided target state prior (bbox in case of SOT), format: xywh
        """
        # use prediction on the last frame as target state prior
        if state is None:
            target_pos_prior, target_sz_prior = self._state['state']
        # use provided bbox as target state prior
        else:
            rect = state  # bbox in xywh format is given for initialization in case of tracking
            box = xywh2cxywh(rect).reshape(4)
            target_pos_prior, target_sz_prior = box[:2], box[2:]
        features = self._state['features']

        # forward inference to estimate new state
        target_pos, target_sz = self.track(im,
                                           target_pos_prior,
                                           target_sz_prior,
                                           features,
                                           update_state=True)

        # save underlying state
        self._state['state'] = target_pos, target_sz

        # return rect format
        track_rect = cxywh2xywh(np.concatenate([target_pos, target_sz],
                                               axis=-1))
        if self._hyper_params["corr_fea_output"]:
            return target_pos, target_sz, self._state["corr_fea"]
        return track_rect

    # ======== tracking processes ======== #

    def _postprocess_score(self, score, online_score, box_wh, target_sz,
                           scale_x):
        r"""
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
        pscore = (
            1 - self._hyper_params["online_score_weight"]
        ) * pscore + self._hyper_params["online_score_weight"] * online_score

        # ipdb.set_trace()
        # cos window (motion model)
        window_influence = self._hyper_params['window_influence']
        pscore = pscore * (
            1 - window_influence) + self._state['window'] * window_influence
        best_pscore_id = np.argmax(pscore)

        return best_pscore_id, pscore, penalty

    def _postprocess_box(self, best_pscore_id, score, box_wh, target_pos,
                         target_sz, scale_x, x_size, penalty):
        r"""
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
        r"""
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
        r"""
        Convert box from cropped patch to original frame
        :param box_in_crop: (4, ), cxywh, box in cropped patch
        :param target_pos: target position
        :param scale_x: scale of cropped patch
        :param x_size: size of cropped patch
        :return:
            box_in_frame: (4, ), cxywh, box in original frame
        """
        x = (box_in_crop[..., 0]) / scale_x + target_pos[0] - (x_size //
                                                               2) / scale_x
        y = (box_in_crop[..., 1]) / scale_x + target_pos[1] - (x_size //
                                                               2) / scale_x
        w = box_in_crop[..., 2] / scale_x
        h = box_in_crop[..., 3] / scale_x
        box_in_frame = np.stack([x, y, w, h], axis=-1)

        return box_in_frame
