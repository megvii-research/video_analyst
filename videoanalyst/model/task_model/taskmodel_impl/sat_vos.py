# -*- coding: utf-8 -*
import torch

from loguru import logger
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_model.taskmodel_base import (TRACK_TASKMODELS,
                                                          VOS_TASKMODELS)
from videoanalyst.utils import md5sum

torch.set_printoptions(precision=8)


@VOS_TASKMODELS.register
class SatVOS(ModuleBase):
    r"""
    State-Aware Tracker model for VOS

    Hyper-Parameters
    ----------------
    pretrain_model_path: string
        path to parameter to be loaded into module
    """

    default_hyper_params = dict(pretrain_model_path="", )

    def __init__(self, GML_extractor, joint_encoder, decoder, loss):
        super(SatVOS, self).__init__()
        self.GML_extractor = GML_extractor
        self.joint_encoder = joint_encoder
        self.decoder = decoder
        # loss
        self.loss = loss

    def forward(self, *args, phase="train"):
        r"""
        Perform VOS process for different phases (e.g. train / global_feature / segment)

        Arguments
        ---------
        filterd_image: torch.Tensor
            filtered image patch for global modeling loop

        saliency_image: torch.Tensor
            saliency image for saliency encoder
        corr_feature: torch.Tensor
            correlated feature produced by siamese encoder
        global_feature: torch.Tensor
            global feature produced by global modeling loop

        Returns
        -------
        f_g: torch.Tensor
            global feature extracted from filtered image
        pred_mask: torch.Tensor
            predicted mask after sigmoid for the patch of saliency image

        """
        # phase: train
        if phase == 'train':
            saliency_image, corr_feature, filtered_image = args
            global_feature = self.GML_extractor(filtered_image)
            enc_features = self.joint_encoder(saliency_image, corr_feature)
            decoder_features = [global_feature] + enc_features
            outputs = self.decoder(decoder_features, phase="train")
            return outputs

        # phase: feature
        elif phase == 'global_feature':
            filtered_image, = args
            f_g = self.GML_extractor(filtered_image)
            out_list = [f_g]
            return out_list

        elif phase == 'segment':
            saliency_image, corr_feature, global_feature = args
            enc_features = self.joint_encoder(saliency_image, corr_feature)
            decoder_features = [global_feature] + enc_features

            outputs = self.decoder(decoder_features, phase="test")
            pred_mask = outputs
            out_list = [pred_mask]
            return out_list

        else:
            raise ValueError("Phase non-implemented.")

    def update_params(self):
        r"""
        Load model parameters
        """
        if self._hyper_params["pretrain_model_path"] != "":
            model_path = self._hyper_params["pretrain_model_path"]
            state_dict = torch.load(model_path,
                                    map_location=torch.device("cpu"))
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            try:
                self.load_state_dict(state_dict, strict=True)
            except:
                self.load_state_dict(state_dict, strict=False)
            logger.info("Pretrained weights loaded from {}".format(model_path))
            logger.info("Check md5sum of Pretrained weights: %s" %
                        md5sum(model_path))

    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)
