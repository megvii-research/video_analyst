r"""
Get root path & root config path
"""
import os.path as osp

ROOT_PATH = osp.dirname(osp.dirname(osp.realpath(__file__)))
ROOT_CFG = osp.join(ROOT_PATH, 'config.yaml')
