# -*- coding: utf-8 -*
import copy
import importlib
import itertools
import math
import os
import os.path as osp
from collections import OrderedDict
from os.path import join
from typing import Dict

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm
from yacs.config import CfgNode

import torch
import torch.multiprocessing as mp

from videoanalyst.evaluation import vot_benchmark
from videoanalyst.utils import ensure_dir

from ..tester_base import TRACK_TESTERS, TesterBase


@TRACK_TESTERS.register
class VOTLTTester(TesterBase):
    r"""
    Tester to test the vot dataset, the result is saved as follows
    exp_dir/logs/$dataset_name$/$tracker_name/
                                    |-baseline/$video_name$/ folder of result files
                                    |-eval_result.csv evaluation result file

    Hyper-parameters
    ----------------
    device_num: int
        number of gpu for test
    data_root: dict
        vot dataset root directory. dict(dataset_name: path_to_root)
    dataset_names: str
        daataset name (VOT2018LT)
    """

    extra_hyper_params = dict(
        device_num=1,
        data_root=CfgNode(dict(VOT2018LT="datasets/")),
        dataset_names=[
            "VOT2018LT",
        ],
        save_video=True,
    )

    def __init__(self, *args, **kwargs):
        r"""
        Crete tester with config and pipeline

        Arguments
        ---------
        cfg: CfgNode
            parent config, (e.g. model / pipeline / tester)
        pipeline: PipelineBase
            pipeline to test
        """
        super(VOTLTTester, self).__init__(*args, **kwargs)
        self._state['speed'] = -1

    def update_params(self):
        pass

    def test(self) -> Dict:
        r"""
        Run test
        """
        # set dir
        self.tracker_name = self._hyper_params["exp_name"]
        test_result_dict = None
        for dataset_name in self._hyper_params["dataset_names"]:
            self.dataset_name = dataset_name
            self.tracker_dir = os.path.join(self._hyper_params["exp_save"],
                                            self.dataset_name)
            self.save_root_dir = os.path.join(self.tracker_dir,
                                              self.tracker_name, "longterm")
            self.save_video_dir = os.path.join(self.tracker_dir,
                                               self.tracker_name, "video")
            ensure_dir(self.save_root_dir)
            ensure_dir(self.save_video_dir)
            # track videos
            self.run_tracker()
            # evaluation
            test_result_dict = self.F1_evaluation()
        return test_result_dict

    def run_tracker(self):
        """
        Run self.pipeline on VOT
        """
        num_gpu = self._hyper_params["device_num"]
        all_devs = [torch.device("cuda:%d" % i) for i in range(num_gpu)]
        logger.info('runing test on devices {}'.format(all_devs))
        vot_root = self._hyper_params["data_root"][self.dataset_name]
        logger.info('Using dataset %s at: %s' % (self.dataset_name, vot_root))
        # setup dataset
        dataset = vot_benchmark.load_dataset(vot_root, self.dataset_name)
        self.dataset = dataset
        keys = list(dataset.keys())
        keys.sort()
        nr_records = len(keys)
        pbar = tqdm(total=nr_records)
        mean_speed = -1
        speed_list = []
        speed_queue = mp.Queue(500)
        # set worker
        if num_gpu == 1:
            self.worker(keys, all_devs[0], speed_queue)
            for i in range(nr_records):
                s = speed_queue.get()
                speed_list.append(s)
                pbar.update(1)
        else:
            nr_video = math.ceil(nr_records / num_gpu)
            procs = []
            for i in range(num_gpu):
                start = i * nr_video
                end = min(start + nr_video, nr_records)
                split_records = keys[start:end]
                proc = mp.Process(target=self.worker,
                                  args=(split_records, all_devs[i],
                                        speed_queue))
                print('process:%d, start:%d, end:%d' % (i, start, end))
                proc.start()
                procs.append(proc)
            for i in range(nr_records):
                s = speed_queue.get()
                speed_list.append(s)
                pbar.update(1)
            for p in procs:
                p.join()
        # print result
        mean_speed = float(np.mean(speed_list))
        logger.info('Mean Speed: {:.2f} FPS'.format(mean_speed))
        self._state['speed'] = mean_speed

    def worker(self, records, dev, speed_queue=None):
        r"""
        Worker to run tracker on records

        Arguments
        ---------
        records:
            specific records, can be a subset of whole sequence
        dev: torch.device object
            target device
        speed_queue:
            queue for fps measurement collecting
        """
        # tracker = copy.deepcopy(self._pipeline)
        tracker = self._pipeline
        tracker.set_device(dev)
        for v_id, video in enumerate(records):
            speed = self.track_single_video(tracker, video, v_id=v_id)
            if speed_queue is not None:
                speed_queue.put_nowait(speed)

    def F1_evaluation(self):
        r"""
        Run evaluation & write result to csv file under self.tracker_dir
        """
        F1Benchmark = importlib.import_module(
            "videoanalyst.evaluation.vot_benchmark.pysot.evaluation",
            package="F1Benchmark").F1Benchmark

        tracker_name = self._hyper_params["exp_name"]
        result_csv = "%s.csv" % tracker_name

        open(join(self.tracker_dir, result_csv), 'a+')
        dataset = vot_benchmark.VOTLTDataset(
            self.dataset_name,
            self._hyper_params["data_root"][self.dataset_name])
        dataset.set_tracker(self.tracker_dir, self.tracker_name)
        f1_benchmark = F1Benchmark(dataset)
        f1_result = {}
        ret = f1_benchmark.eval(self.tracker_name)
        f1_result.update(ret)
        f1_benchmark.show_result(f1_result)
        test_result_dict = dict()
        test_result_dict["main_performance"] = 0
        return test_result_dict

    def track_single_video(self, tracker, video, v_id=0):
        r"""
        track frames in single video with VOT rules

        Arguments
        ---------
        tracker: PipelineBase
            pipeline
        video: str
            video name
        v_id: int
            video id
        """
        vot_float2str = importlib.import_module(
            "videoanalyst.evaluation.vot_benchmark.pysot.utils.region",
            package="vot_float2str").vot_float2str
        regions = []
        scores = []
        video = self.dataset[video]
        image_files, gt = video['image_files'], video['gt']
        start_frame, end_frame, toc = 0, len(image_files), 0
        vw = None

        for f, image_file in enumerate(tqdm(image_files)):
            im = vot_benchmark.get_img(image_file)
            im_show = im.copy().astype(np.uint8)
            if self._hyper_params["save_video"] and vw is None:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                video_path = os.path.join(self.save_video_dir,
                                          video['name'] + ".avi")
                width, height = im.shape[1], im.shape[0]
                vw = cv2.VideoWriter(video_path, fourcc, 25,
                                     (int(width), int(height)))
            tic = cv2.getTickCount()
            if f == start_frame:  # init
                cx, cy, w, h = vot_benchmark.get_axis_aligned_bbox(gt[f])
                location = vot_benchmark.cxy_wh_2_rect((cx, cy), (w, h))
                tracker.init(im, location)
                regions.append(1)
                scores.append(None)
            elif f > start_frame:  # tracking
                location = tracker.update(im)
                regions.append(location)
                scores.append(tracker._state["pscore"])
            toc += cv2.getTickCount() - tic
            if self._hyper_params["save_video"]:
                cv2.rectangle(im_show, (int(location[0]), int(location[1])),
                              (int(location[0] + location[2]),
                               int(location[1] + location[3])), (255, 0, 0), 2)
                cv2.putText(im_show, str(scores[-1]), (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                vw.write(im_show)
        if vw is not None:
            vw.release()

        toc /= cv2.getTickFrequency()

        # save result
        result_dir = join(self.save_root_dir, video['name'])
        ensure_dir(result_dir)
        result_path = join(result_dir, '{:s}_001.txt'.format(video['name']))
        with open(result_path, "w") as fin:
            for x in regions:
                fin.write("{:d}\n".format(x)) if isinstance(x, int) else \
                    fin.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')
        result_path = os.path.join(
            result_dir, '{}_001_confidence.value'.format(video['name']))
        with open(result_path, 'w') as fin:
            for x in scores:
                fin.write('\n') if x is None else fin.write(
                    "{:.6f}\n".format(x))
        logger.info(
            '({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}'.format(
                v_id, video['name'], toc, f / toc))

        return f / toc

    def write_result_to_csv(self,
                            ar_result,
                            eao_result,
                            speed=-1,
                            param=None,
                            result_csv=None):
        write_header = (osp.getsize(result_csv.name) == 0)
        row_dict = OrderedDict()
        row_dict['tracker'] = self.tracker_name
        row_dict['speed'] = speed

        ret = ar_result[self.tracker_name]
        overlaps = list(itertools.chain(*ret['overlaps'].values()))
        accuracy = np.nanmean(overlaps)
        length = sum([len(x) for x in ret['overlaps'].values()])
        failures = list(ret['failures'].values())
        lost_number = np.mean(np.sum(failures, axis=0))
        robustness = np.mean(np.sum(np.array(failures), axis=0) / length) * 100
        eao = eao_result[self.tracker_name]['all']

        row_dict['dataset'] = self.dataset_name
        row_dict['accuracy'] = accuracy
        row_dict['robustness'] = robustness
        row_dict['lost'] = lost_number
        row_dict['eao'] = eao

        if write_header:
            header = ','.join([str(k) for k in row_dict.keys()])
            result_csv.write('%s\n' % header)
        row_data = ','.join([str(v) for v in row_dict.values()])
        result_csv.write('%s\n' % row_data)


VOTLTTester.default_hyper_params = copy.deepcopy(
    VOTLTTester.default_hyper_params)
VOTLTTester.default_hyper_params.update(VOTLTTester.extra_hyper_params)
