# -*- coding: utf-8 -*
import argparse
import os.path as osp
import time

import cv2
from loguru import logger

import torch

from videoanalyst.config.config import cfg, specify_task
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.pipeline.utils.bbox import xywh2xyxy
from videoanalyst.utils.image import ImageFileVideoStream, ImageFileVideoWriter
from videoanalyst.utils.visualization import VideoWriter

font_size = 0.5
font_width = 1


def make_parser():
    parser = argparse.ArgumentParser(
        description="press s to select the target box,\n \
                        then press enter or space to confirm it or press c to cancel it,\n \
                        press c to stop track and press q to exit program")
    parser.add_argument(
        "-cfg",
        "--config",
        default="experiments/siamfcpp/test/got10k/siamfcpp_alexnet-got.yaml",
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
    parser.add_argument("-i",
                        "--init-bbox",
                        type=float,
                        nargs="+",
                        default=[-1.0],
                        help="initial bbox, length=4, format=xywh")
    return parser


def main(args):
    root_cfg = cfg
    root_cfg.merge_from_file(args.config)
    logger.info("Load experiment configuration at: %s" % args.config)

    # resolve config
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()
    window_name = task_cfg.exp_name
    # build model
    model = model_builder.build(task, task_cfg.model)
    # build pipeline
    pipeline = pipeline_builder.build(task, task_cfg.pipeline, model)
    dev = torch.device(args.device)
    pipeline.set_device(dev)
    init_box = None
    template = None
    if len(args.init_bbox) == 4:
        init_box = args.init_bbox

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
            logger.debug("frame: {}".format(frame_idx))
            if template is not None:
                time_a = time.time()
                rect_pred = pipeline.update(frame)
                logger.debug(rect_pred)
                show_frame = frame.copy()
                time_cost = time.time() - time_a
                bbox_pred = xywh2xyxy(rect_pred)
                bbox_pred = tuple(map(int, bbox_pred))
                cv2.putText(show_frame,
                            "track cost: {:.4f} s".format(time_cost), (128, 20),
                            cv2.FONT_HERSHEY_COMPLEX, font_size, (0, 0, 255),
                            font_width)
                cv2.rectangle(show_frame, bbox_pred[:2], bbox_pred[2:],
                              (0, 255, 0))
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
        if (init_box is None) or (vw is None):
            logger.debug("Press key s to select object.")
            if (frame_idx == 0):
                wait_time = 5000
            else:
                wait_time = 30
            key = cv2.waitKey(wait_time) & 0xFF
        logger.debug("key: {}".format(key))
        if key == ord("q"):
            break
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        elif key == ord("s"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            logger.debug("Select object to track")
            box = cv2.selectROI(window_name,
                                frame,
                                fromCenter=False,
                                showCrosshair=True)
            if box[2] > 0 and box[3] > 0:
                init_box = box
        elif key == ord("c"):
            logger.debug(
                "init_box/template released, press key s again to select object."
            )
            init_box = None
            template = None
        if (init_box is not None) and (template is None):
            template = cv2.resize(
                frame[int(init_box[1]):int(init_box[1] + init_box[3]),
                      int(init_box[0]):int(init_box[0] + init_box[2])],
                (128, 128))
            pipeline.init(frame, init_box)
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
