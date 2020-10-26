# -*- coding: utf-8 -*
import argparse
import hashlib
import json
import os

from loguru import logger

from videoanalyst.evaluation.got_benchmark.utils.ioutils import (download,
                                                                 extract)


def download_vot(version, root_dir):
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    elif os.path.isfile(os.path.join(root_dir, 'list.txt')):
        with open(os.path.join(root_dir, 'list.txt')) as f:
            seq_names = f.read().strip().split('\n')
        if all([os.path.isdir(os.path.join(root_dir, s)) for s in seq_names]):
            logger.warning('Files already downloaded.')
            return

    url = 'http://data.votchallenge.net/'
    if version.startswith('LT'):
        # long-term tracking challenge
        year = int(version[2:])
        homepage = url + 'vot{}/longterm/'.format(year)
    elif version.startswith('RGBD'):
        # RGBD tracking challenge
        year = int(version[4:])
        homepage = url + 'vot{}/rgbd/'.format(year)
    elif version.startswith('RGBT'):
        # RGBT tracking challenge
        year = int(version[4:])
        url = url + 'vot{}/rgbtir/'.format(year)
        homepage = url + 'meta/'
    elif int(version) in range(2013, 2015 + 1):
        # main challenge (2013~2015)
        homepage = url + 'vot{}/dataset/'.format(version)
    elif int(version) in range(2015, 2019 + 1):
        # main challenge (2016~2019)
        homepage = url + 'vot{}/main/'.format(version)

    # download description file
    bundle_url = homepage + 'description.json'
    bundle_file = os.path.join(root_dir, 'description.json')
    if not os.path.isfile(bundle_file):
        logger.info('Downloading description file from {}'.format(bundle_url))
        download(bundle_url, bundle_file)

    # read description file
    logger.info('\nParsing description file {}'.format(bundle_file))
    with open(bundle_file) as f:
        bundle = json.load(f)

    # md5 generator
    def md5(filename):
        hash_md5 = hashlib.md5()
        with open(filename, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    # download all sequences
    seq_names = []
    for seq in bundle['sequences']:
        seq_name = seq['name']
        seq_names.append(seq_name)

        # download channel (color/depth/ir) files
        channels = seq['channels'].keys()
        seq_files = []
        for cn in channels:
            seq_url = seq['channels'][cn]['url']
            if not seq_url.startswith(('http', 'https')):
                seq_url = url + seq_url[seq_url.find('sequence'):]
            seq_file = os.path.join(root_dir, '{}_{}.zip'.format(seq_name, cn))
            if not os.path.isfile(seq_file) or \
                md5(seq_file) != seq['channels'][cn]['checksum']:
                logger.info('\nDownloading {} from {}'.format(
                    seq_name, seq_url))
                download(seq_url, seq_file)
            seq_files.append(seq_file)

        # download annotations
        anno_url = homepage + '%s.zip' % seq_name
        anno_file = os.path.join(root_dir, seq_name + '_anno.zip')
        if not os.path.isfile(anno_file) or \
            md5(anno_file) != seq['annotations']['checksum']:
            download(anno_url, anno_file)

        # unzip compressed files
        seq_dir = os.path.join(root_dir, seq_name)
        if not os.path.isfile(seq_dir) or len(os.listdir(seq_dir)) < 10:
            logger.info('Extracting %s...' % seq_name)
            os.makedirs(seq_dir)
            for seq_file in seq_files:
                extract(seq_file, seq_dir)
            extract(anno_file, seq_dir)

    # save list.txt
    list_file = os.path.join(root_dir, 'list.txt')
    with open(list_file, 'w') as f:
        f.write(str.join('\n', seq_names))

        return root_dir


def make_parser():
    parser = argparse.ArgumentParser(
        description="download vot datasets from official webset")
    parser.add_argument(
        "dataset_name",
        help=
        'dataset name in [2016, 2018, 2019, LT2018, LT2019, RGBD2019, RGBT2019]'
    )
    parser.add_argument("dataset_root", help="path to save dataset")
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    logger.info("start downloading {} to {}".format(args.dataset_name,
                                                    args.dataset_root))
    download_vot(args.dataset_name, args.dataset_root)
