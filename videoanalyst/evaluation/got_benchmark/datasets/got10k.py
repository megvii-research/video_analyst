from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
import pickle
import six

from tqdm import tqdm

_VALID_SUBSETS = ['train', 'val', 'test']


class GOT10k(object):
    r"""`GOT-10K <http://got-10k.aitestunion.com//>`_ Dataset.

    Publication:
        ``GOT-10k: A Large High-Diversity Benchmark for Generic Object
        Tracking in the Wild``, L. Huang, X. Zhao and K. Huang, ArXiv 2018.
    
    Args:
        root_dir (string): Root directory of dataset where ``train``,
            ``val`` and ``test`` folders exist.
        subset (string, optional): Specify ``train``, ``val`` or ``test``
            subset of GOT-10k.
        return_meta (string, optional): If True, returns ``meta``
            of each sequence in ``__getitem__`` function, otherwise
            only returns ``img_files`` and ``anno``.
        list_file (string, optional): If provided, only read sequences
            specified by the file instead of all sequences in the subset.
    """
    data_dict = {subset : dict() for subset in _VALID_SUBSETS}
    def __init__(self, root_dir, subset='test', return_meta=False,
                 list_file=None, check_integrity=True, cache_path=None, ignore_cache=False):
        super(GOT10k, self).__init__()
        assert subset in _VALID_SUBSETS, 'Unknown subset.'
        self.root_dir = root_dir
        self.subset = subset
        self.return_meta = False if subset == 'test' else return_meta

        if list_file is None:
            list_file = os.path.join(root_dir, subset, 'list.txt')
        if check_integrity:
            self._check_integrity(root_dir, subset, list_file)

        with open(list_file, 'r') as f:
            self.seq_names = f.read().strip().split('\n')

        # check if subset cache already exists in cls.data_dict
        if (self.subset in GOT10k.data_dict) and (len(GOT10k.data_dict[self.subset]) > 0):
            return

        # load subset cache into cls.data_dict
        if cache_path is None:
            cache_path = os.path.join(root_dir, subset+".pkl")
        if os.path.isfile(cache_path) and not ignore_cache:
            print("cache file {} exists".format(cache_path))
            with open(cache_path, "rb") as f:
                GOT10k.data_dict[self.subset] = pickle.load(f)
            print("loaded cache file {}".format(cache_path))
        # build subset cache in cls.data_dict and cache to storage
        else:
            print("start loading got-10k {}".format(subset))
            for seq_name in tqdm(self.seq_names):
                seq_dir = os.path.join(root_dir, subset, seq_name)
                img_files, anno, meta = self.load_single_sequence(seq_dir)
                GOT10k.data_dict[self.subset][seq_name] = dict(img_files = img_files, anno=anno, meta=meta)
            with open(cache_path, "wb") as f:
                pickle.dump(GOT10k.data_dict[self.subset], f)
            print("dump cache file to {}".format(cache_path))



    def load_single_sequence(self, seq_dir):
        img_files = sorted(glob.glob(os.path.join(
            seq_dir, '*.jpg')))
        anno = np.loadtxt(os.path.join(seq_dir, "groundtruth.txt"), delimiter=',')

        if self.subset == 'test' and anno.ndim == 1:
            assert len(anno) == 4
            anno = anno[np.newaxis, :]
        else:
            assert len(img_files) == len(anno)

        if self.return_meta or self.subset == "val":
            meta = self._fetch_meta(seq_dir)
            return img_files, anno, meta
        else:
            return img_files, anno, None
    
    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple: (img_files, anno) if ``return_meta`` is False, otherwise
                (img_files, anno, meta), where ``img_files`` is a list of
                file names, ``anno`` is a N x 4 (rectangles) numpy array, while
                ``meta`` is a dict contains meta information about the sequence.
        """
        if isinstance(index, int):
            seq_name = self.seq_names[index]
        else:
            if not index in self.seq_names:
                print('Sequence {} not found.'.format(index))
                print(len(self.seq_names))
                raise Exception('Sequence {} not found.'.format(index))
            seq_name = index
        img_files = GOT10k.data_dict[self.subset][seq_name]["img_files"]
        anno = GOT10k.data_dict[self.subset][seq_name]["anno"]

        if self.subset == 'test' and (anno.size // 4 == 1):
            anno = anno.reshape(-1, 4)
            # anno = anno[np.newaxis, :]
        else:
            assert len(img_files) == len(anno)

        if self.return_meta:
            meta = GOT10k.data_dict[self.subset][seq_name]["meta"]
            return img_files, anno, meta
        else:
            return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _check_integrity(self, root_dir, subset, list_file=None):
        assert subset in ['train', 'val', 'test']
        if list_file is None:
            list_file = os.path.join(root_dir, subset, 'list.txt')

        if os.path.isfile(list_file):
            with open(list_file, 'r') as f:
                seq_names = f.read().strip().split('\n')
            
            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, subset, seq_name)
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            # dataset not exists
            raise Exception('Dataset not found or corrupted.')

    def _fetch_meta(self, seq_dir):
        # meta information
        meta_file = os.path.join(seq_dir, 'meta_info.ini')
        with open(meta_file) as f:
            meta = f.read().strip().split('\n')[1:]
        meta = [line.split(': ') for line in meta]
        meta = {line[0]: line[1] for line in meta}

        # attributes
        attributes = ['cover', 'absence', 'cut_by_image']
        for att in attributes:
            meta[att] = np.loadtxt(os.path.join(seq_dir, att + '.label'))

        return meta
