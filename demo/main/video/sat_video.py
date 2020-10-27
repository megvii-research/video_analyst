# -*- coding: utf-8 -*
import argparse
import os.path as osp
import time

import cv2
import numpy as np
from loguru import logger

import torch

from videoanalyst.config.config import cfg, specify_task
from videoanalyst.engine.monitor.monitor_impl.utils import (labelcolormap,
                                                            mask_colorize)
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.utils.image import ImageFileVideoStream, ImageFileVideoWriter
from videoanalyst.utils.visualization import VideoWriter

font_size = 0.5
font_width = 1
color_map = labelcolormap(10)
polygon_points = []
lbt_flag = False
rbt_flag = False


def draw_polygon(event, x, y, flags, param):
    global polygon_points, lbt_flag, rbt_flag
    if event == cv2.EVENT_LBUTTONDOWN:
        lbt_flag = True
        polygon_points.append((x, y))  # 用于画点
    if event == cv2.EVENT_RBUTTONDOWN:
        rbt_flag = True


def make_parser():
    parser = argparse.ArgumentParser(
        description=
        "press s to select the target mask, left click for new pt, right click to finish,\n \
                        then press enter or space to confirm it or press c to cancel it,\n \
                        press c to stop track and press q to exit program")
    parser.add_argument("-cfg",
                        "--config",
                        default="experiments/sat/test/sat_res50-davis17.yaml",
                        type=str,
                        help='experiment configuration')
    parser.add_argument("-d",
                        "--device",
                        default="cpu",
                        type=str,
                        help="torch.device, cuda or cpu")
    parser.add_argument(
        "-v",
        "--video",
        type=str,
        default="webcam",
        help=
        r"video input mode. \"webcam\" for webcamera, \"path/*.<extension>\" for image files, \"path/file.<extension>\". Default is webcam. "
    )
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        default="",
                        help="path to dump the track video")
    parser.add_argument("-s",
                        "--start-index",
                        type=int,
                        default=0,
                        help="start index / #frames to skip")
    parser.add_argument(
        "-r",
        "--resize",
        type=float,
        default=1.0,
        help="resize result image to anothor ratio (for saving bandwidth)")
    parser.add_argument(
        "-do",
        "--dump-only",
        action="store_true",
        help=
        "only dump, do not show image (in cases where cv2.imshow inccurs errors)"
    )
    return parser


def main(args):
    global polygon_points, lbt_flag, rbt_flag
    root_cfg = cfg
    root_cfg.merge_from_file(args.config)
    logger.info("Load experiment configuration at: %s" % args.config)

    # resolve config
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()
    window_name = task_cfg.exp_name
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_polygon)
    # build model
    tracker_model = model_builder.build("track", task_cfg.tracker_model)
    tracker = pipeline_builder.build("track",
                                     task_cfg.tracker_pipeline,
                                     model=tracker_model)
    segmenter = model_builder.build('vos', task_cfg.segmenter)
    # build pipeline
    pipeline = pipeline_builder.build('vos',
                                      task_cfg.pipeline,
                                      segmenter=segmenter,
                                      tracker=tracker)
    dev = torch.device(args.device)
    pipeline.set_device(dev)
    init_mask = None
    init_box = None
    template = None

    video_name = "untitled"
    vw = None
    resize_ratio = args.resize
    dump_only = args.dump_only

    # create video stream
    # from webcam
    if args.video == "webcam":
        logger.info("Starting video stream...")
        vs = cv2.VideoCapture(0)
        vs.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        formated_time_str = time.strftime(r"%Y%m%d-%H%M%S", time.localtime())
        video_name = "webcam-{}".format(formated_time_str)
    # from image files
    elif not osp.isfile(args.video):
        logger.info("Starting from video frame image files...")
        vs = ImageFileVideoStream(args.video, init_counter=args.start_index)
        video_name = osp.basename(osp.dirname(args.video))
    # from video file
    else:
        logger.info("Starting from video file...")
        vs = cv2.VideoCapture(args.video)
        video_name = osp.splitext(osp.basename(args.video))[0]

    # create video writer to output video
    if args.output:
        # save as image files
        if not str(args.output).endswith(r".mp4"):
            vw = ImageFileVideoWriter(osp.join(args.output, video_name))
        # save as a single video file
        else:
            vw = VideoWriter(args.output, fps=20)

    # loop over sequence
    frame_idx = 0  # global frame index
    while vs.isOpened():
        key = 255
        ret, frame = vs.read()
        if ret:
            if template is not None:
                time_a = time.time()
                score_map = pipeline.update(frame)
                mask = (score_map > 0.5).astype(np.uint8) * 2
                color_mask = mask_colorize(mask, 10, color_map)
                color_mask = cv2.resize(color_mask,
                                        (frame.shape[1], frame.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                show_frame = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)
                time_cost = time.time() - time_a
                cv2.putText(show_frame,
                            "track cost: {:.4f} s".format(time_cost), (128, 20),
                            cv2.FONT_HERSHEY_COMPLEX, font_size, (0, 0, 255),
                            font_width)
                if template is not None:
                    show_frame[:128, :128] = template
            else:
                show_frame = frame
            show_frame = cv2.resize(
                show_frame, (int(show_frame.shape[1] * resize_ratio),
                             int(show_frame.shape[0] * resize_ratio)))  # resize
            if not dump_only:
                cv2.imshow(window_name, show_frame)
            if vw is not None:
                vw.write(show_frame)
        else:
            break
        # catch key if
        if (init_mask is None) or (vw is None):
            if (frame_idx == 0):
                wait_time = 5000
            else:
                wait_time = 30
            key = cv2.waitKey(wait_time) & 0xFF
        if key == ord("q"):
            break
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        elif key == ord("s"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            logger.debug(
                "Select points object to track, left click for new pt, right click to finish"
            )
            polygon_points = []
            while not rbt_flag:
                if lbt_flag:
                    print(polygon_points[-1])
                    cv2.circle(show_frame, polygon_points[-1], 5, (0, 0, 255),
                               2)
                    if len(polygon_points) > 1:
                        cv2.line(show_frame, polygon_points[-2],
                                 polygon_points[-1], (255, 0, 0), 2)
                    lbt_flag = False
                cv2.imshow(window_name, show_frame)
                key = cv2.waitKey(10) & 0xFF
            if len(polygon_points) > 2:
                np_pts = np.array(polygon_points)
                init_box = cv2.boundingRect(np_pts)
                zero_mask = np.zeros((show_frame.shape[0], show_frame.shape[1]),
                                     dtype=np.uint8)
                init_mask = cv2.fillPoly(zero_mask, [np_pts], (1, ))
            rbt_flag = False
        elif key == ord("c"):
            logger.debug(
                "init_box/template released, press key s again to select object."
            )
            init_mask = None
            init_box = None
            template = None
        if (init_mask is not None) and (template is None):
            template = cv2.resize(
                frame[int(init_box[1]):int(init_box[1] + init_box[3]),
                      int(init_box[0]):int(init_box[0] + init_box[2])],
                (128, 128))
            pipeline.init(frame, init_box, init_mask)
            logger.debug("pipeline initialized with bbox : {}".format(init_box))
        frame_idx += 1

    vs.release()
    if vw is not None:
        vw.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(args)
