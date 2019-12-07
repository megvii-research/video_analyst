# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------

from __future__ import division
# from config import config
# from tracker import tracker_config
import math
from multiprocessing import Queue, Process
import argparse
import logging
import glob
import numpy as np
import cv2
import os
import os.path as osp
from os import makedirs
from os.path import join, isdir
import sys
import itertools
from collections import OrderedDict

import torch

logger = logging.getLogger('global')

# from evaluation.tracking.vot_benchmark.log_helper import init_log, add_file_handler
# from evaluation.tracking.vot_benchmark.bbox_helper import get_axis_aligned_bbox, cxy_wh_2_rect
# from evaluation.tracking.vot_benchmark.benchmark_helper import load_dataset, get_img
# from evaluation.tracking.vot_benchmark.pysot.utils.region import vot_overlap, vot_float2str
# from evaluation.tracking.vot_benchmark.pysot.datasets import VOTDataset
# from evaluation.tracking.vot_benchmark.pysot.evaluation import AccuracyRobustnessBenchmark, EAOBenchmark

from tqdm import tqdm

root_path = osp.dirname(osp.dirname(osp.realpath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# from IPython import embed;embed()

from videoanalyst.evaluation import vot_benchmark
from videoanalyst.evaluation.utils import build_tracker_wt_model
from videoanalyst.utils import load_cfg
from videoanalyst.utils import ensure

# get video common config
basic_cfg_path = osp.join(root_path, 'config.yaml')
config = load_cfg(basic_cfg_path)
tracker_cfg_path = osp.join(root_path,
                            'experiments/siamfc++/siamfcpp_alexnet.yaml')
tracker_cfg = load_cfg(tracker_cfg_path)
# from IPython import embed;embed()

# def track_vot(tracker, tracker_name, tracker_config, video, v_id=0):
def track_vot(tracker, tracker_name, video, v_id=0):

    result_path = join(config.logs.vot_dir, tracker_name, 'baseline', video['name'])
    # lib_repo.ensure_dir(result_path)
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
            # location = vot_benchmark.cxy_wh_2_rect(tracker.state['target_pos'], tracker.state['target_sz'])
            location = vot_benchmark.cxy_wh_2_rect((cx, cy), (w, h))
            # tracker.init(im, np.array([cx, cy, w, h]), tracker_config)
            # tracker.init(im, location, config=tracker_config)
            tracker.init(im, location)
            regions.append(1 if 'VOT' in args.dataset else gt[f])
            gt_polygon = None
            pred_polygon = None
        elif f > start_frame:  # tracking
            location = tracker.update(im)

            # cv2.rectangle(im_show,
            #               (int(location[0]), int(location[1])),
            #               (int(location[0]+location[2]), int(location[1]+location[3])),
            #               (0, 255, 255))
            # cv2.imshow('debug', im_show)
            # cv2.waitKey(0)

            gt_polygon = (gt[f][0], gt[f][1], gt[f][2], gt[f][3],
                          gt[f][4], gt[f][5], gt[f][6], gt[f][7])
            pred_polygon = (location[0], location[1], location[0] + location[2], location[1],
                            location[0] + location[2], location[1] + location[3], location[0],
                            location[1] + location[3])
            # from IPython import embed;embed()
            b_overlap = vot_benchmark.vot_overlap(gt_polygon, pred_polygon, (im.shape[1], im.shape[0]))
            gt_polygon = ((gt[f][0], gt[f][1]), (gt[f][2], gt[f][3]), (gt[f][4], gt[f][5]), (gt[f][6], gt[f][7]))
            pred_polygon = ((location[0], location[1]), (location[0] + location[2], location[1]),
                            (location[0] + location[2], location[1] + location[3]), (location[0], location[1] + location[3]))

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
    # lib_repo.ensure_dir(result_path)
    ensure(result_path)
    result_path = join(result_path, '{:s}_001.txt'.format(video['name']))
    with open(result_path, "w") as fin:
        for x in regions:
            fin.write("{:d}\n".format(x)) if isinstance(x, int) else \
                fin.write(','.join([vot_benchmark.vot_float2str("%.4f", i) for i in x]) + '\n')

    logger.info('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps Lost: {:d} '.format(
        v_id, video['name'], toc, f / toc, lost_times))

    return lost_times, f / toc


# def load_model(model_file, dev, tracker_config):
#     """
#
#     :param model_file:
#     :param dev:
#     :param config:
#     :return:
#     """
#     # snapshot = torch.load(model_file, map_location=dev)
#     model = SiamFCOT(config=tracker_config).to(dev)
#     state_dict = lib_repo.load_state_dict(model_file, dev=dev)
#     model.load_state_dict(state_dict, strict=False)
#     model.eval()
#     return model

# def worker(model_file, tracker_name, tracker_config, dev, records, dataset, result_queue=None, speed_queue=None):
def worker(tracker_name, tracker_cfg, dev, records, dataset, result_queue=None, speed_queue=None):
    # model = load_model(model_file, dev, tracker_config)
    # tracker = SiamRPNTracker(model, device=dev)
    tracker = build_tracker_wt_model(tracker_cfg, dev)
    for v_id, video in enumerate(records):
        # lost, speed = track_vot(tracker, tracker_name, tracker_config, dataset[video], v_id=v_id)
        lost, speed = track_vot(tracker, tracker_name, dataset[video], v_id=v_id)
        if result_queue is not None:
            result_queue.put_nowait(lost)
        if speed_queue is not None:
            speed_queue.put_nowait(speed)


def write_result_to_csv(tracker_names, dataset, ar_result, eao_result, speed=-1, param=None, result_csv=None):
    write_header = (os.path.getsize(result_csv.name)==0)

    for tracker_name in tracker_names:
        row_dict = OrderedDict()

        row_dict['tracker'] = tracker_name
        row_dict['speed'] = speed
        row_dict.update(param)

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
            result_csv.write('%s\n'%header)

        row_data = ','.join([str(v) for v in row_dict.values()])
        result_csv.write('%s\n' % row_data)



def evaluation(tracker_dir, tracker_name, param_dict, mean_speed=-1):
    # nr_param_settings = sum([len(paramlist) for paramlist in paramlist_dict.values()])
    global args
    # if nr_param_settings>1:
    if args.mode.lower() == 'param':
        file_to_write = open(join(tracker_dir, tracker_name, args.result_file), 'a+')
        csv_to_write = open(join(tracker_dir, tracker_name, args.result_csv), 'a+')
    else:
        file_to_write = open(join(tracker_dir, args.result_file), 'a+')
        csv_to_write = open(join(tracker_dir, args.result_csv), 'a+')
    trackers = glob.glob(join(tracker_dir, tracker_name))
    trackers = [t.split('/')[-1] for t in trackers]
    assert len(trackers) > 0
    dataset = vot_benchmark.VOTDataset(args.dataset, args.data_root)
    dataset.set_tracker(tracker_dir, trackers)
    ar_benchmark = vot_benchmark.AccuracyRobustnessBenchmark(dataset)
    ar_result = {}
    from multiprocessing import Pool
    with Pool(processes=min(5, len(trackers))) as pool:
        for ret in tqdm(pool.imap_unordered(ar_benchmark.eval, trackers), desc='eval ar', total=len(trackers), ncols=100):
            ar_result.update(ret)
    benchmark = vot_benchmark.EAOBenchmark(dataset)
    eao_result = {}
    with Pool(processes=min(5, len(trackers))) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval, trackers), desc='eval eao', total=len(trackers), ncols=100):
            eao_result.update(ret)
    ar_benchmark.show_result(ar_result, eao_result, show_video_level=args.show_video_level)
    print_decimals = 4
    for k, v in param_dict.items():
        param_dict[k] = np.round(v, decimals=print_decimals)
    file_to_write.write('FPS={:.2f}, {}\n'.format(mean_speed, param_dict))
    ar_benchmark.write_result(ar_result, eao_result, show_video_level=args.show_video_level, result_file=file_to_write)

    write_result_to_csv(trackers, args.dataset, ar_result, eao_result, speed=mean_speed, param=param_dict, result_csv=csv_to_write)


def main():
    global args, logger, v_id, paramlist_dict

    nr_gpu = 1
    all_devs = [torch.device("cuda:%d" % i) for i in range(nr_gpu)]
    print('all_devs', all_devs)

    #dataset
    print('Using dataset: %s'%args.dataset)
    # vot2018_root = config.vot2018_root
    vot2018_root = config.data.VOT2018.path
    # vot2019_root = config.vot2019_root
    vot2019_root = config.data.VOT2019.path
    if args.dataset == 'VOT2018':
        vot_root = vot2018_root
        args.data_root = vot_root
    elif args.dataset == 'VOT2019':
        vot_root = vot2019_root
        args.data_root = vot_root
    else:
        raise ValueError('Invalid dataset name %s', args.dataset)
    # setup dataset
    dataset = vot_benchmark.load_dataset(vot_root, args.dataset)
    keys = list(dataset.keys())
    keys.sort()
    nr_records = len(keys)
    pbar = tqdm(total=nr_records)

    # for param_settings in itertools.product(*(paramlist_dict.values())):
    #     param_dict = OrderedDict([(param_name, param_var) for param_name, param_var in zip(paramlist_dict.keys(), param_settings)])
    #     # tracker_config.update(param_dict)
    #     print('*'*10 + 'search for settings:', param_dict, '*'*10)
    #     for epoch in range(args.start_epoch, args.end_epoch+1):
            # if args.model:
            #     model_file = args.model
            # else:
            #     model_file = join(config.snapshot_dir, 'epoch-' + str(epoch) + '.pkl')
            # if not model_file.startswith("s3://"):
            #     model_file = osp.realpath(model_file)
            # print("model_file:", model_file)
            # tracker_name = lib_repo.extract_fname(model_file)
    tracker_name = tracker_cfg.track.exp_name
    mean_speed = -1
    # if args.inference:
    vot_benchmark.init_log('global', logging.INFO)
    if args.log != "":
        # log_dir = join(config.vot_dir, 'epoch-' + str(epoch))
        log_dir = join(config.logs.vot_dir, 'test')
        # lib_repo.ensure_dir(log_dir)
        ensure(log_dir)
        vot_benchmark.add_file_handler('global', join(log_dir, args.log), logging.INFO)
    print(log_dir)
    logger = logging.getLogger('global')
    logger.info(args)

    total_lost = 0
    speed_list = []
    nr_devs = len(all_devs)
    result_queue = Queue(500)
    speed_queue = Queue(500)
    # from IPython import embed;embed()
    if nr_devs == 1:
        # worker(model_file, tracker_name, tracker_config, all_devs[0], keys, dataset, result_queue, speed_queue)
        worker(tracker_name, tracker_cfg, all_devs[0], keys, dataset, result_queue, speed_queue)

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
            # proc = Process(target=worker,
            #                args=(model_file, tracker_name, tracker_config, all_devs[i], split_records, dataset, result_queue, speed_queue))
            proc = Process(target=worker,
                           args=(tracker_name, tracker_cfg, all_devs[i], split_records, dataset, result_queue, speed_queue))
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
    evaluation(config.logs.vot_dir, tracker_name, param_dict, mean_speed=mean_speed)

def make_parser():
    parser = argparse.ArgumentParser(description='Test SiamRPN')
    parser.add_argument('-l', '--log', default="log_test.txt", type=str, help='log file')
    # parser.add_argument('-d', '--devices', default='cpu0')
    parser.add_argument('--mode', default='epoch')
    parser.add_argument('--model', '-m', default='')
    parser.add_argument('--start_epoch', '-se', default=0, type=int)
    parser.add_argument('--end_epoch', '-ee', default=-1, type=int)
    parser.add_argument('--dataset', dest='dataset', default='VOT2018', help='datasets')
    parser.add_argument('--data_root', default=config.data.VOT2018.path, type=str, metavar='PATH', help='path to VOT2018.json')
    # parser.add_argument('--tracker_prefix', default='epoch-', type=str, metavar='PATH', help='path to VOT2018.json')
    parser.add_argument('--result_file', default='search_epoch.txt', type=str, help='save epoch-level result')
    parser.add_argument('--result_csv', default='search_epoch.csv', type=str, help='save epoch-level result into csv')
    parser.add_argument('--show_video_level', action='store_true')

    parser.add_argument('--config', default='', type=str, help='experiment configuration')

    return parser

param_arg_prefix = '--param_'
linspace_suffix = '_linspace'
arange_suffix = '_arange'
def parse_unknowns_args(unknowns):
    param_name_idxs = [i for i, name in enumerate(unknowns) if name.startswith(param_arg_prefix)]
    param_name_idxs.append(len(unknowns))
    slice_idxs = list(zip(*[param_name_idxs[:-1], param_name_idxs[1:]]))
    paramlist_dict = OrderedDict()
    # from IPython import embed;embed()
    for (idx1, idx2) in slice_idxs:
        slice = unknowns[idx1:idx2]
        param_name = slice[0][len(param_arg_prefix):]
        values = [int(float(s)) if float(s).is_integer() else float(s) for s in slice[1:]]

        if param_name.endswith(linspace_suffix):
            paramlist_dict[param_name[:-len(linspace_suffix)]] = np.linspace(*values[:3]).tolist()
        elif param_name.endswith(arange_suffix):
            paramlist_dict[param_name[:-len(arange_suffix)]] = np.arange(*values[:3]).tolist()
        else:
            paramlist_dict[param_name] = values
    return paramlist_dict


if __name__ == '__main__':
    parser = make_parser()
    args, unknowns = parser.parse_known_args()
    paramlist_dict = parse_unknowns_args(unknowns)
    for param_settings in itertools.product(*(paramlist_dict.values())):
        param_dict = {param_name: param_var for param_name, param_var in zip(paramlist_dict.keys(), param_settings)}
    # from IPython import embed;embed()

# preprocess the command
    # test model file
    if args.model:
        args.end_epoch = args.start_epoch = -1
        print('Model: %s' % args.model)
    # test only one epoch
    else:
        if args.end_epoch == -1:
            args.end_epoch = args.start_epoch
        print('Start epoch: %d'%args.start_epoch)
        print('End epoch: %d'%args.end_epoch)
    # show param search range
    if args.mode.lower() == 'epoch':
        print('*'*20, 'Search epoch', '*'*20)
    elif args.mode.lower() == 'param':
        print('*'*20, 'Grid search range', '*'*20)
        for k, v in paramlist_dict.items():
            print(k, v)
        print('*'*20, 'Grid search range', '*'*20)
    else:
        raise ValueError('Invalid mode name %s', args.mode)

    main()

