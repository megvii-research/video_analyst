# -*- coding: utf-8 -*

import numpy as np
import torch
import cv2
import torch.nn as nn

from copy import deepcopy
from videoanalyst.pipeline.pipeline_base import PipelineBase
from videoanalyst.pipeline.pipeline_base import VOS_PIPELINES
from videoanalyst.pipeline.utils import (cxywh2xywh, get_crop,
                                         get_subwindow_tracking,
                                         imarray_to_tensor, tensor_to_numpy,
                                         xywh2cxywh, xyxy2cxywh)


# ============================== Tracker definition ============================== #
@VOS_PIPELINES.register
class StateAwareTracker(PipelineBase):
    r"""
    Basic State-Aware Tracker

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

        track_ema: bool
            adopt template updating strategy for tracker
        track_ema_u: float
            hyper-parameter u for tracker's template updating
        track_ema_s: float
            hyper-parameter s for tracker's template updating

        save_patch: bool
            save and visualize the predicted mask for saliency image patch
        mask_pred_thresh: float
            threshold to binarize predicted mask for final decision
        mask_filter_thresh: float
            threshold to binarize predicted mask for filter the patch of global modeling loop
        GMP_image_size: int
            image size of the input of global modeling loop
        saliency_image_size: int
            image size of saliency image
        saliency_image_field: int
            corresponding fields of saliency image
        cropping_strategy: bool
            use cropping strategy or not
        state_score_thresh: float
            threshhold for state score
        global_modeling: bool
            use global modeling loop or not
        seg_ema_u: float
            hyper-parameter u for global feature updating
        seg_ema_s: float
            hyper-parameter s for global feature updating


    Hyper-parameters (to be calculated at runtime)
    ----------------------------------------------
    score_size: int
        final feature map
    score_offset: int
        final feature map
    """
    default_hyper_params = dict(
        # hyper-parameters for siamfc++
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
        track_ema=True,
        track_ema_u=0.1,
        track_ema_s=0.6,

        # hyper-parameters for state-aware tracker
        save_patch=True,
        mask_pred_thresh=0.4,
        mask_filter_thresh=0.5,
        GMP_image_size=129,
        saliency_image_size=257,
        saliency_image_field=129,
        cropping_strategy=True,
        state_score_thresh=0.85,
        global_modeling=True,
        seg_ema_u=0.5,
        seg_ema_s=0.5,
    )

    def __init__(self, segmenter, tracker):

        self._hyper_params = deepcopy(
            self.default_hyper_params)  # mapping-like object
        self._state = dict()  # pipeline state
        self._segmenter = segmenter
        self._tracker = tracker

        self.update_params()

        # set underlying model to device
        self.device = torch.device("cpu")
        self.debug = False
        self.set_model(self._segmenter, self._tracker)

    def set_model(self, segmenter, tracker):
        """model to be set to pipeline. change device & turn it into eval mode
        
        Parameters
        ----------
        model : ModuleBase
            model to be set to pipeline
        """
        self._segmenter = segmenter.to(self.device)
        self._segmenter.eval()
        self._tracker = tracker.to(self.device)
        self._tracker.eval()

    def set_device(self, device):
        self.device = device
        self._segmenter = self._segmenter.to(device)
        self._tracker = self._tracker.to(device)

    def update_params(self):
        hps = self._hyper_params
        hps['score_size'] = (
            hps['x_size'] -
            hps['z_size']) // hps['total_stride'] + 1 - hps['num_conv3x3'] * 2
        hps['score_offset'] = (
            hps['x_size'] - 1 -
            (hps['score_size'] - 1) * hps['total_stride']) // 2
        self._hyper_params = hps

    def track_feature(self, im, target_pos, target_sz, avg_chans=None):
        r"""
        Extract target image feature for tracker
        :param im: image frame
        :param target_pos: target position (x, y)
        :param target_sz: target size (w, h)
        :param avg_chans: channel mean values
        :return f_z feature of target image
        :return im_z_crop cropped patch of target image
        :return avg_chans channel average
        """
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
            f_z = self._tracker(imarray_to_tensor(im_z_crop).to(self.device),
                                phase=phase)[0]

        return f_z, im_z_crop, avg_chans

    def init(self, im, state, init_mask):
        """
        initialize the whole pipeline :
        tracker init => global modeling loop init

        :param im: init frame
        :param state: bbox in xywh format
        :param init_mask: binary mask of target object in shape (h,w)
        """

        #========== SiamFC++ init ==============

        rect = state  # bbox in xywh format is given for initialization in case of tracking
        box = xywh2cxywh(rect)
        target_pos, target_sz = box[:2], box[2:]

        self._state['im_h'] = im.shape[0]
        self._state['im_w'] = im.shape[1]

        # extract template feature
        f_z, im_z_crop, avg_chans = self.track_feature(im, target_pos,
                                                       target_sz)

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
        self._state['f_z'] = f_z
        self._state['window'] = window
        self._state['state'] = (target_pos, target_sz)
        if self._hyper_params['track_ema']:
            self._state['f_ema_track'] = f_z
            self._state['f_init_track'] = f_z

        # ========== Global Modeling Loop init ==============
        init_mask_c3 = np.stack([init_mask, init_mask, init_mask],
                                -1).astype(np.uint8)
        init_mask_crop_c3, _ = get_crop(
            init_mask_c3,
            target_pos,
            target_sz,
            z_size=self._hyper_params["z_size"],
            x_size=self._hyper_params["GMP_image_size"],
            avg_chans=avg_chans * 0,
            context_amount=0.5,
            func_get_subwindow=get_subwindow_tracking,
        )
        init_mask_crop = init_mask_crop_c3[:, :, 0]
        init_mask_crop = (init_mask_crop >
                          self._hyper_params['mask_filter_thresh']).astype(
                              np.uint8)
        init_mask_crop = np.expand_dims(init_mask_crop,
                                        axis=-1)  #shape: (129,129,1)

        init_image, _ = get_crop(
            im,
            target_pos,
            target_sz,
            z_size=127,
            x_size=129,
            avg_chans=avg_chans,
            context_amount=0.5,
            func_get_subwindow=get_subwindow_tracking,
        )

        #self._state['prev_mask'] = init_mask_crop #shape: (129,129,1)
        #self._state['prev_image'] = init_image #shape: (129,129,3)
        #print(init_mask_crop.shape, init_image.shape)
        filtered_image = init_mask_crop * init_image
        self._state['filtered_image'] = filtered_image  #shape: (129,129,3)

        with torch.no_grad():
            deep_feature = self._segmenter(imarray_to_tensor(filtered_image).to(
                self.device),
                                           phase='global_feature')[0]

        self._state['seg_init_feature'] = deep_feature  #shape : (1,256,5,5)
        self._state['seg_global_feature'] = deep_feature
        self._state['gml_feature'] = deep_feature
        self._state['conf_score'] = 1

    def track4vos(self,
                  im_x,
                  target_pos,
                  target_sz,
                  f_z,
                  update_state=False,
                  **kwargs):
        r"""
        similarity encoder with regression head
        returns regressed bbox and correlation feature

        :param im_x: current frame
        :param target_pos: target position (x, y)
        :param target_sz: target size (w, h)
        :param f_z: target feature
        :return new_target_pos, new_target_sz, corr_feature
        """

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
        with torch.no_grad():
            score, box, cls, ctr, corr_feature = self._tracker(
                imarray_to_tensor(im_x_crop).to(self.device),
                f_z,
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
        self._state['current_state'] = (target_pos, target_sz)
        self._state['scale_x'] = scale_x

        # record optional mid-level info
        if update_state:
            self._state['score'] = score
            self._state['pscore'] = pscore
            self._state['all_box'] = box
            self._state['cls'] = cls
            self._state['ctr'] = ctr

        return new_target_pos, new_target_sz, corr_feature

    def track_template_updation(self, im, target_pos, target_sz):
        r"""
        template updation for tracking,  based on confidence score

        :param im: frame for target feature extraction
        :param target_pos: target position (x, y)
        :param target_sz: target size (w, h)
        """

        u = self._hyper_params['track_ema_u']
        s = self._hyper_params['track_ema_s']
        conf_score = self._state['conf_score']
        f_ema_track = self._state['f_ema_track']
        f_init_track = self._state['f_init_track']

        u = u * conf_score
        feature, im_z_crop, avg_chans = self.track_feature(
            im, target_pos, target_sz)

        f_ema_track = f_ema_track * (1 - u) + feature * u
        f_z = f_ema_track * s + f_init_track * (1 - s)
        self._state['f_z'] = f_z

    def global_modeling(self):
        """
        always runs after seg4vos, takes newly predicted filtered image,
        extracts high-level feature and updates the global feature based on confidence score

        """
        filtered_image = self._state['filtered_image']  # shape: (129,129,3)
        with torch.no_grad():
            deep_feature = self._segmenter(imarray_to_tensor(filtered_image).to(
                self.device),
                                           phase='global_feature')[0]

        seg_global_feature = self._state['seg_global_feature']
        seg_init_feature = self._state['seg_init_feature']
        u = self._hyper_params['seg_ema_u']
        s = self._hyper_params['seg_ema_s']
        conf_score = self._state['conf_score']

        u = u * conf_score
        seg_global_feature = seg_global_feature * (1 - u) + deep_feature * u
        gml_feature = seg_global_feature * s + seg_init_feature * (1 - s)

        self._state['seg_global_feature'] = seg_global_feature
        self._state['gml_feature'] = gml_feature

    def joint_segmentation(self, im_x, target_pos, target_sz, corr_feature,
                           gml_feature, **kwargs):
        r"""
        segment the current frame for VOS
        crop image => segmentation =>  params updation

        :param im_x: current image
        :param target_pos: target position (x, y)
        :param target_sz: target size (w, h)
        :param corr_feature: correlated feature produced by siamese encoder
        :param gml_feature: global feature produced by gloabl modeling loop
        :return: pred_mask  mask prediction in the patch of saliency image
        :return: pred_mask_b binary mask prediction in the patch of saliency image
        """

        if 'avg_chans' in kwargs:
            avg_chans = kwargs['avg_chans']
        else:
            avg_chans = self._state['avg_chans']

        # crop image for saliency encoder
        saliency_image, _ = get_crop(
            im_x,
            target_pos,
            target_sz,
            z_size=127,
            x_size=257,
            avg_chans=avg_chans,
            context_amount=0.5,
            func_get_subwindow=get_subwindow_tracking,
        )

        # mask prediction
        pred_mask = self._segmenter(imarray_to_tensor(saliency_image).to(
            self.device),
                                    corr_feature,
                                    gml_feature,
                                    phase='segment')[0]  #tensor(1,1,257,257)

        pred_mask = tensor_to_numpy(pred_mask[0]).transpose(
            (1, 2, 0))  #np (257,257,1)

        # post processing
        mask_filter = (pred_mask >
                       self._hyper_params['mask_filter_thresh']).astype(
                           np.uint8)
        pred_mask_b = (pred_mask >
                       self._hyper_params['mask_pred_thresh']).astype(np.uint8)

        if self._hyper_params['save_patch']:
            mask_red = np.zeros_like(saliency_image)
            mask_red[:, :, 0] = mask_filter[:, :, 0] * 255
            masked_image = saliency_image * 0.5 + mask_red * 0.5
            self._state['patch_prediction'] = masked_image

        filtered_image = saliency_image * mask_filter
        filtered_image = cv2.resize(filtered_image, (129, 129))
        self._state['filtered_image'] = filtered_image

        try:
            conf_score = (pred_mask * pred_mask_b).sum() / pred_mask_b.sum()
        except:
            conf_score = 0
        self._state['conf_score'] = conf_score

        mask_in_full_image = self._mask_back(pred_mask)
        self._state['mask_in_full_image'] = mask_in_full_image  # > 0.5

        return pred_mask, pred_mask_b

    def cropping_strategy(self, p_mask_b, track_pos, track_size):
        r"""
        swithes the bbox prediction strategy based on the estimation of predicted mask.
        returns newly predicted target position and size

        :param p_mask_b: binary mask prediction in the patch of saliency image
        :param target_pos: target position (x, y)
        :param target_sz: target size (w, h)
        :return: new_target_pos, new_target_sz
        """

        new_target_pos, new_target_sz = track_pos, track_size
        conf_score = self._state['conf_score']

        if conf_score > self._hyper_params['state_score_thresh']:
            contours, _ = cv2.findContours(p_mask_b, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            cnt_area = [cv2.contourArea(cnt) for cnt in contours]

            if len(contours) != 0 and np.max(cnt_area) > 10:
                contour = contours[np.argmax(cnt_area)]  # use max area polygon
                polygon = contour.reshape(-1, 2)
                pbox = cv2.boundingRect(polygon)  # Min Max Rectangle  x1,y1,w,h
                rect_full, cxywh_full = self._coord_back(pbox)
                mask_pos, mask_sz = cxywh_full[:2], cxywh_full[2:]

                conc_score = np.max(cnt_area) / sum(cnt_area)
                state_score = conf_score * conc_score
                self._state['conc_score'] = conc_score
                self._state['state_score'] = state_score

                if state_score > self._hyper_params['state_score_thresh']:
                    new_target_pos, new_target_sz = mask_pos, mask_sz

                # rect_in_full, cxywh_in_full = coor_back(pbox)
                self._state['mask_rect'] = rect_full

            else:  # empty mask
                self._state['mask_rect'] = [-1, -1, -1, -1]
                self._state['state_score'] = 0

        else:  # empty mask
            self._state['mask_rect'] = [-1, -1, -1, -1]
            self._state['state_score'] = 0
        return new_target_pos, new_target_sz

    def update(self, im):

        # get track
        # target_pos_prior, target_sz_prior = self.state['target_pos'], self.state['target_sz']
        target_pos_prior, target_sz_prior = self._state['state']
        f_z = self._state['f_z']

        # forward inference to estimate new state
        # tracking for VOS returns regressed box and correlation feature
        target_pos_track, target_sz_track, corr_feature = self.track4vos(
            im, target_pos_prior, target_sz_prior, f_z, update_state=True)

        # segmentation returnd predicted masks
        gml_feature = self._state['gml_feature']
        pred_mask, pred_mask_b = self.joint_segmentation(
            im, target_pos_prior, target_sz_prior, corr_feature, gml_feature)

        # template updation for tracker
        if self._hyper_params['track_ema']:
            self.track_template_updation(im, target_pos_track, target_sz_track)

        # global modeling loop updates global feature for next frame's segmentation
        if self._hyper_params['global_modeling']:
            self.global_modeling()

        # cropping strategy loop swtiches the coordinate prediction method
        if self._hyper_params['cropping_strategy']:
            target_pos, target_sz = self.cropping_strategy(
                pred_mask_b, target_pos_track, target_sz_track)
        else:
            target_pos, target_sz = target_pos_track, target_sz_track

        # save underlying state
        self._state['state'] = target_pos, target_sz

        # return rect format
        track_rect = cxywh2xywh(
            np.concatenate([target_pos_track, target_sz_track], axis=-1))
        return track_rect

    # ======== vos processes ======== #

    def _mask_back(self, p_mask, size=257, region=129):
        """
        Warp the predicted mask from cropped patch back to original image.

        :param p_mask: predicted_mask (h,w)
        :param size: image size of cropped patch
        :param region: region size with template = 127
        :return: mask in full image
        """

        target_pos, target_sz = self._state['current_state']
        scale_x = self._state['scale_x']

        zoom_ratio = size / region
        scale = scale_x * zoom_ratio
        cx_f, cy_f = target_pos[0], target_pos[1]
        cx_c, cy_c = (size - 1) / 2, (size - 1) / 2

        a, b = 1 / (scale), 1 / (scale)

        c = cx_f - a * cx_c
        d = cy_f - b * cy_c

        mapping = np.array([[a, 0, c], [0, b, d]]).astype(np.float)
        mask_in_full_image = cv2.warpAffine(
            p_mask,
            mapping, (self._state['im_w'], self._state['im_h']),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0)
        # print(scale_z, a, b )
        return mask_in_full_image

    def _coord_back(self, rect, size=257, region=129):
        """
        Warp the predicted coordinates from cropped patch back to original image.

        :param rect: rect with coords in cropped patch
        :param size: image size of cropped patch
        :param region: region size with template = 127
        :return: rect(xywh) and cxywh in full image
        """

        target_pos, target_sz = self._state['current_state']
        scale_x = self._state['scale_x']

        zoom_ratio = size / region
        scale = scale_x * zoom_ratio
        cx_f, cy_f = target_pos[0], target_pos[1]
        cx_c, cy_c = (size - 1) / 2, (size - 1) / 2

        a, b = 1 / (scale), 1 / (scale)

        c = cx_f - a * cx_c
        d = cy_f - b * cy_c

        x1, y1, w, h = rect[0], rect[1], rect[2], rect[3]

        x1_t = a * x1 + c
        y1_t = b * y1 + d
        w_t, h_t = w * a, h * b
        return [x1_t, y1_t, w_t, h_t], xywh2cxywh([x1_t, y1_t, w_t, h_t])

    # ======== tracking processes ======== #

    def _postprocess_score(self, score, box_wh, target_sz, scale_x):
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
