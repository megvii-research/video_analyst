# -*- coding: utf-8 -*
import copy
import itertools
import logging
import math
import os
from collections import OrderedDict
from multiprocessing import Process, Queue
from os.path import join

import numpy as np
from tqdm import tqdm

import torch

import cv2
from videoanalyst.engine.tester.tester_base import TRACK_TESTERS, TesterBase
from videoanalyst.evaluation import vot_benchmark
from videoanalyst.utils import ensure_dir

vot_benchmark.init_log('global', logging.INFO)
logger = logging.getLogger("global")


@TRACK_TESTERS.register
class VOTTester(TesterBase):
    r"""
    Tester to test the vot dataset, the result is saved as follows
    exp_dir/logs/$dataset_name$/$tracker_name$/baseline
                                    |-$video_name$/ floder of result files        
                                    |-eval_result.csv evaluation result file

    """

    default_hyper_params = dict(
        device_num=1,
        vot_data_root={
            "VOT2018": "datasets/VOT/vot2018",
            "VOT2019": "datasets/VOT/vot2019"
        },
        dataset_names=[
            "VOT2018",
        ],
    )

    def __init__(self, cfg, pipeline):
        """
        Crete tester with config and pipeline

        Arguments
        ---------
        cfg: CfgNode
            parent config, (e.g. model / pipeline / tester)
        pipeline: PipelineBase
            pipeline to test
        """
        super(VOTTester, self).__init__(cfg, pipeline)
        self._state['speed'] = -1

    def test(self):
        """
        Run test
        """
        # set dir
        self.tracker_name = self._cfg.exp_name
        for dataset_name in self._hyper_params["dataset_names"]:
            self.dataset_name = dataset_name
            # self.tracker_dir = os.path.join(self._cfg.auto.log_dir, self._hyper_params["dataset_name"])
            self.tracker_dir = os.path.join(self._cfg.exp_save,
                                            self.dataset_name)
            self.save_root_dir = os.path.join(self.tracker_dir,
                                              self.tracker_name, "baseline")
            ensure_dir(self.save_root_dir)
            # track videos
            self.run_tracker()
            # evaluation
            self.evaluation()

    def run_tracker(self):
        """
        Run self.pipeline on VOT
        """
        num_gpu = self._hyper_params["device_num"]
        all_devs = [torch.device("cuda:%d" % i) for i in range(num_gpu)]
        logging.info('runing test on devices {}'.format(all_devs))
        vot_root = self._hyper_params["vot_data_root"][self.dataset_name]
        logger.info('Using dataset %s at: %s' % (self.dataset_name, vot_root))
        # setup dataset
        dataset = vot_benchmark.load_dataset(vot_root, self.dataset_name)
        self.dataset = dataset
        keys = list(dataset.keys())
        keys.sort()
        nr_records = len(keys)
        pbar = tqdm(total=nr_records)
        mean_speed = -1
        total_lost = 0
        speed_list = []
        result_queue = Queue(500)
        speed_queue = Queue(500)
        # set worker
        if num_gpu == 1:
            self.worker(keys, all_devs[0], result_queue, speed_queue)
            for i in range(nr_records):
                t = result_queue.get()
                s = speed_queue.get()
                total_lost += t
                speed_list.append(s)
                pbar.update(1)
        else:
            nr_video = math.ceil(nr_records / num_gpu)
            procs = []
            for i in range(num_gpu):
                start = i * nr_video
                end = min(start + nr_video, nr_records)
                split_records = keys[start:end]
                proc = Process(target=self.worker,
                               args=(split_records, all_devs[i], result_queue,
                                     speed_queue))
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
        # print result
        mean_speed = float(np.mean(speed_list))
        logger.info('Total Lost: {:d}'.format(total_lost))
        logger.info('Mean Speed: {:.2f} FPS'.format(mean_speed))
        self._state['speed'] = mean_speed

    def worker(self, records, dev, result_queue=None, speed_queue=None):
        """
        Worker to run tracker on records

        Arguments
        ---------
        records:
            specific records, can be a subset of whole sequence
        dev: torch.device object
            target device
        result_queue:
            queue for result collecting
        speed_queue:
            queue for fps measurement collecting
        """
        tracker = copy.deepcopy(self._pipeline)
        tracker.to_device(dev)
        for v_id, video in enumerate(records):
            lost, speed = self.track_single_video(tracker, video, v_id=v_id)
            if result_queue is not None:
                result_queue.put_nowait(lost)
            if speed_queue is not None:
                speed_queue.put_nowait(speed)

    def evaluation(self):
        """
        Run evaluation & write result to csv file under self.tracker_dir
        """
        tracker_name = self._cfg.exp_name
        result_csv = "%s.csv" % tracker_name

        csv_to_write = open(join(self.tracker_dir, result_csv), 'a+')
        dataset = vot_benchmark.VOTDataset(
            self.dataset_name,
            self._hyper_params["vot_data_root"][self.dataset_name])
        dataset.set_tracker(self.tracker_dir, self.tracker_name)
        ar_benchmark = vot_benchmark.AccuracyRobustnessBenchmark(dataset)
        ar_result = {}
        ret = ar_benchmark.eval(self.tracker_name)
        ar_result.update(ret)
        ar_benchmark.show_result(ar_result)
        benchmark = vot_benchmark.EAOBenchmark(dataset)
        eao_result = {}
        ret = benchmark.eval(self.tracker_name)
        eao_result.update(ret)
        ar_benchmark.show_result(ar_result,
                                 eao_result=eao_result,
                                 show_video_level=False)
        self.write_result_to_csv(
            ar_result,
            eao_result,
            speed=self._state['speed'],
            result_csv=csv_to_write,
        )
        csv_to_write.close()

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
        regions = []
        video = self.dataset[video]
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
                regions.append(1 if 'VOT' in self.dataset_name else gt[f])
                gt_polygon = None
                pred_polygon = None
            elif f > start_frame:  # tracking
                location = tracker.update(im)

                gt_polygon = (gt[f][0], gt[f][1], gt[f][2], gt[f][3], gt[f][4],
                              gt[f][5], gt[f][6], gt[f][7])
                pred_polygon = (location[0], location[1],
                                location[0] + location[2], location[1],
                                location[0] + location[2],
                                location[1] + location[3], location[0],
                                location[1] + location[3])
                b_overlap = vot_benchmark.vot_overlap(
                    gt_polygon, pred_polygon, (im.shape[1], im.shape[0]))
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
        result_dir = join(self.save_root_dir, video['name'])
        ensure_dir(result_dir)
        result_path = join(result_dir, '{:s}_001.txt'.format(video['name']))
        with open(result_path, "w") as fin:
            for x in regions:
                fin.write("{:d}\n".format(x)) if isinstance(x, int) else \
                    fin.write(','.join([vot_benchmark.vot_float2str("%.4f", i) for i in x]) + '\n')

        logger.info(
            '({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps Lost: {:d} '
            .format(v_id, video['name'], toc, f / toc, lost_times))

        return lost_times, f / toc

    def write_result_to_csv(self,
                            ar_result,
                            eao_result,
                            speed=-1,
                            param=None,
                            result_csv=None):
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

        header = ','.join([str(k) for k in row_dict.keys()])
        result_csv.write('%s\n' % header)
        row_data = ','.join([str(v) for v in row_dict.values()])
        result_csv.write('%s\n' % row_data)
