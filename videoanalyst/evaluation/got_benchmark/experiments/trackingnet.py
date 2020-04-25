from __future__ import absolute_import

import os
import numpy as np

from loguru import logger

from .otb import ExperimentOTB
from ..datasets import TrackingNet
from ..utils.metrics import rect_iou, center_error


class ExperimentTrackingNet(ExperimentOTB):
    r"""Experiment pipeline and evaluation toolkit for TrackingNet dataset.
       Only the TEST subset part implemented.

    Args:
        root_dir (string): Root directory of LaSOT dataset.
        subset (string, optional): Specify ``train`` or ``test`` or ``train0,1,...``
            subset of TrackingNet.  Default is ``test``.
        return_meta (bool, optional): whether to fetch meta info
        (occlusion or out-of-view).  Default is ``False``.
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """
    def __init__(self, root_dir, subset='test', return_meta=False,
                 result_dir='results', report_dir='reports'):
        # assert subset.upper() in ['TRAIN', 'TEST']
        assert subset.startswith(('train', 'test')), 'Unknown subset.'
        self.dataset = TrackingNet(root_dir, subset, return_meta=return_meta)
        self.result_dir = os.path.join(result_dir, 'TrackingNet')
        self.report_dir = os.path.join(report_dir, 'TrackingNet')


        # as nbins_iou increases, the success score
        # converges to the average overlap (AO)
        self.nbins_iou = 21
        self.nbins_ce = 51

    def report(self, tracker_names, *args, plot_curves=True, **kwargs):
        try:
            performance = super(ExperimentTrackingNet, self).report(tracker_names, *args, plot_curves=plot_curves, **kwargs)
        except:
            logger.info("report error, performance=None")
            report_dir = os.path.join(self.report_dir, tracker_names[0])
            performance = None

        return performance


    # def _calc_metrics(self, boxes, anno):
    #     valid = ~np.any(np.isnan(anno), axis=1)
    #     if len(valid) == 0:
    #         print('Warning: no valid annotations')
    #         return None, None
    #     else:
    #         ious = rect_iou(boxes[valid, :], anno[valid, :])
    #         center_errors = center_error(
    #             boxes[valid, :], anno[valid, :])
    #         return ious, center_errors
