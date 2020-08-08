# -*- coding: utf-8 -*
import os
import os.path as osp

import cv2
from loguru import logger


class VideoWriter(object):
    """
    Video writer which handles video recording overhead
    Usage:
        object creation: provide path to write
        write:
        release:
    """
    def __init__(self, video_file, fps=25, scale=1.0):
        """

        :param video_file: path to write video. Perform nothing in case of None
        :param fps: frame per second
        :param scale: resize scale
        """
        self.video_file = video_file
        self.fps = fps
        self.writer = None
        self.scale = scale

    def write(self, frame):
        """

        :param frame: numpy array, (H, W, 3), BGR, frame to write
        :return:
        """
        h, w = frame.shape[:2]
        h_rsz, w_rsz = int(h * self.scale), int(w * self.scale)
        frame = cv2.resize(frame, (w_rsz, h_rsz))
        if self.writer is None:
            video_dir = osp.dirname(osp.realpath(self.video_file))
            if not osp.exists(video_dir):
                os.makedirs(video_dir)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.video_file, fourcc, self.fps,
                                          tuple(frame.shape[1::-1]))
        self.writer.write(frame)

    def release(self):
        """
        Manually release
        :return:
        """
        if self.writer is None:
            return
        self.writer.release()
        logger.info("video file dumped at {}".format(self.video_file))

    def __del__(self):
        self.release()
