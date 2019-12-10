# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------

from __future__ import division

import glob
import itertools
import logging
# from config import config
# from tracker import tracker_config
import math
import os
import os.path as osp
import sys
from collections import OrderedDict
from multiprocessing import Process, Queue
from os.path import join

import cv2
import numpy as np
from tqdm import tqdm

import torch

from videoanalyst.engine.tester.tester_base import TESTERS
from videoanalyst.evaluation import vot_benchmark
from videoanalyst.evaluation.utils import build_tracker_wt_model
from videoanalyst.utils import ensure

logger = logging.getLogger('global')

root_path = osp.dirname(osp.dirname(osp.realpath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)


def track_vot(tracker, exp_cfg, video, v_id=0):
    tracker_name = exp_cfg.test.exp_name
    result_path = join(exp_cfg.test.exp_save, tracker_name, 'baseline',
                       video['name'])
    ensure(result_path)

    regions = []
    image_files, gt = video['image_files'], video['gt']
    start_frame, end_frame, lost_times, toc = 0, len(image_files), 0, 0
    for f, image_file in enumerate(tqdm(image_files)):
        im = vot_benchmark.get_img(image_file)
        im_show = im.copy().astype(np.uint8)

        tic = cv2.getTickCount()
        if f == start_frame:  # init
            cx, cy, w, h = vot_benchmark.get_axis_aligned_bbox(gt[f])
            location = vot_benchmark.cxy_wh_2_rect((cx, cy), (w, h))
            tracker.init(im, location)
            regions.append(1 if 'VOT' in exp_cfg.test.dataset else gt[f])
            gt_polygon = None
            pred_polygon = None
        elif f > start_frame:  # tracking
            location = tracker.update(im)

            gt_polygon = (gt[f][0], gt[f][1], gt[f][2], gt[f][3], gt[f][4],
                          gt[f][5], gt[f][6], gt[f][7])
            pred_polygon = (location[0], location[1], location[0] + location[2],
                            location[1], location[0] + location[2],
                            location[1] + location[3], location[0],
                            location[1] + location[3])
            b_overlap = vot_benchmark.vot_overlap(gt_polygon, pred_polygon,
                                                  (im.shape[1], im.shape[0]))
            gt_polygon = ((gt[f][0], gt[f][1]), (gt[f][2], gt[f][3]),
                          (gt[f][4], gt[f][5]), (gt[f][6], gt[f][7]))
            pred_polygon = ((location[0], location[1]),
                            (location[0] + location[2],
                             location[1]), (location[0] + location[2],
                                            location[1] + location[3]),
                            (location[0], location[1] + location[3]))

            if b_overlap:
                regions.append(location)
            else:  # lost
                regions.append(2)
                lost_times += 1
                start_frame = f + 5  # skip 5 frames
        else:  # skip
            regions.append(0)
        toc += cv2.getTickCount() - tic

    toc /= cv2.getTickFrequency()

    # save result
    ensure(result_path)
    result_path = join(result_path, '{:s}_001.txt'.format(video['name']))
    with open(result_path, "w") as fin:
        for x in regions:
            fin.write("{:d}\n".format(x)) if isinstance(x, int) else \
                fin.write(','.join([vot_benchmark.vot_float2str("%.4f", i) for i in x]) + '\n')

    logger.info(
        '({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps Lost: {:d} '.
        format(v_id, video['name'], toc, f / toc, lost_times))

    return lost_times, f / toc


def worker(exp_cfg, dev, records, dataset, result_queue=None, speed_queue=None):
    tracker = build_tracker_wt_model(exp_cfg.track, dev)
    for v_id, video in enumerate(records):
        lost, speed = track_vot(tracker, exp_cfg, dataset[video], v_id=v_id)
        if result_queue is not None:
            result_queue.put_nowait(lost)
        if speed_queue is not None:
            speed_queue.put_nowait(speed)


def write_result_to_csv(tracker_names,
                        dataset,
                        ar_result,
                        eao_result,
                        speed=-1,
                        param=None,
                        result_csv=None):
    write_header = (os.path.getsize(result_csv.name) == 0)

    for tracker_name in tracker_names:
        row_dict = OrderedDict()

        row_dict['tracker'] = tracker_name
        row_dict['speed'] = speed

        ret = ar_result[tracker_name]
        overlaps = list(itertools.chain(*ret['overlaps'].values()))
        accuracy = np.nanmean(overlaps)
        length = sum([len(x) for x in ret['overlaps'].values()])
        failures = list(ret['failures'].values())
        lost_number = np.mean(np.sum(failures, axis=0))
        robustness = np.mean(np.sum(np.array(failures), axis=0) / length) * 100
        eao = eao_result[tracker_name]['all']

        row_dict['dataset'] = dataset
        row_dict['accuracy'] = accuracy
        row_dict['robustness'] = robustness
        row_dict['lost'] = lost_number
        row_dict['eao'] = eao

        if write_header:
            header = ','.join([str(k) for k in row_dict.keys()])
            result_csv.write('%s\n' % header)

        row_data = ','.join([str(v) for v in row_dict.values()])
        result_csv.write('%s\n' % row_data)


def evaluation(common_cfg, exp_cfg, mean_speed=-1):
    tracker_dir = exp_cfg.test.exp_save
    tracker_name = exp_cfg.test.exp_name
    dataset_name = exp_cfg.test.dataset
    result_csv = "%s.csv" % tracker_name

    csv_to_write = open(join(exp_cfg.test.exp_save, result_csv), 'a+')
    trackers = glob.glob(join(tracker_dir, tracker_name))
    trackers = [t.split('/')[-1] for t in trackers]
    assert len(trackers) > 0
    dataset = vot_benchmark.VOTDataset(dataset_name,
                                       common_cfg.dataset[dataset_name].path)
    dataset.set_tracker(tracker_dir, trackers)
    ar_benchmark = vot_benchmark.AccuracyRobustnessBenchmark(dataset)
    ar_result = {}
    from multiprocessing import Pool
    with Pool(processes=min(5, len(trackers))) as pool:
        for ret in tqdm(pool.imap_unordered(ar_benchmark.eval, trackers),
                        desc='eval ar',
                        total=len(trackers),
                        ncols=100):
            ar_result.update(ret)
    benchmark = vot_benchmark.EAOBenchmark(dataset)
    eao_result = {}
    with Pool(processes=min(5, len(trackers))) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval, trackers),
                        desc='eval eao',
                        total=len(trackers),
                        ncols=100):
            eao_result.update(ret)
    show_video_level = False
    ar_benchmark.show_result(ar_result,
                             eao_result,
                             show_video_level=show_video_level)
    write_result_to_csv(trackers,
                        dataset_name,
                        ar_result,
                        eao_result,
                        speed=mean_speed,
                        result_csv=csv_to_write)


@TESTERS.register
def VOT2018(*args, **kwargs):
    # parsed_args.dataset = 'VOT2018'
    VOT(*args, **kwargs)


@TESTERS.register
def VOT2019(parsed_args, *args, **kwargs):
    # parsed_args.dataset = 'VOT2019'
    VOT(*args, **kwargs)


@TESTERS.register
def VOT(parsed_args, common_cfg, exp_cfg, *args, **kwargs):
    num_gpu = exp_cfg.test.num_gpu
    all_devs = [torch.device("cuda:%d" % i) for i in range(num_gpu)]
    print('all_devs', all_devs)

    if not exp_cfg.test.exp_save:
        exp_cfg.test.exp_save = common_cfg.logs.vot_dir

    dataset_name = exp_cfg.test.dataset
    vot_root = common_cfg.dataset[dataset_name].path

    logger.info('Using dataset %s at: %s' % (dataset_name, vot_root))

    # setup dataset
    dataset = vot_benchmark.load_dataset(vot_root, dataset_name)
    keys = list(dataset.keys())
    keys.sort()
    nr_records = len(keys)
    pbar = tqdm(total=nr_records)

    mean_speed = -1
    vot_benchmark.init_log('global', logging.INFO)

    total_lost = 0
    speed_list = []
    nr_devs = len(all_devs)
    result_queue = Queue(500)
    speed_queue = Queue(500)
    if nr_devs == 1:
        worker(exp_cfg, all_devs[0], keys, dataset, result_queue, speed_queue)

        for i in range(nr_records):
            t = result_queue.get()
            s = speed_queue.get()
            total_lost += t
            speed_list.append(s)
            pbar.update(1)
    else:
        nr_video = math.ceil(nr_records / nr_devs)
        procs = []
        for i in range(nr_devs):
            start = i * nr_video
            end = min(start + nr_video, nr_records)
            split_records = keys[start:end]
            proc = Process(target=worker,
                           args=(exp_cfg, all_devs[i], split_records, dataset,
                                 result_queue, speed_queue))
            print('process:%d, start:%d, end:%d' % (i, start, end))
            proc.start()
            procs.append(proc)
        for i in range(nr_records):
            t = result_queue.get()
            s = speed_queue.get()
            total_lost += t
            speed_list.append(s)
            pbar.update(1)
        for p in procs:
            p.join()
    mean_speed = float(np.mean(speed_list))
    logger.info('Total Lost: {:d}'.format(total_lost))
    logger.info('Mean Speed: {:.2f} FPS'.format(mean_speed))
    evaluation(common_cfg, exp_cfg, mean_speed=mean_speed)
