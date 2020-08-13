# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "siamrpn_r50_l234_dwxcorr"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Anchor Target
# Positive anchor threshold
__C.TRAIN.THR_HIGH = 0.6

# Negative anchor threshold
__C.TRAIN.THR_LOW = 0.3

# Number of negative
__C.TRAIN.NEG_NUM = 16

# Number of positive
__C.TRAIN.POS_NUM = 16

# Number of anchors per images
__C.TRAIN.TOTAL_NUM = 64

__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 255

__C.TRAIN.BASE_SIZE = 8

__C.TRAIN.OUTPUT_SIZE = 25

__C.TRAIN.RESUME = ''

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 32

__C.TRAIN.NUM_WORKERS = 1

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 1.2

__C.TRAIN.MASK_WEIGHT = 1

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# Random shift see [SiamPRN++](https://arxiv.org/pdf/1812.11703)
# for detail discussion
__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

# Sample Negative pair see [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
# for detail discussion
__C.DATASET.NEG = 0.2

# improve tracking performance for otb100
__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('VID', 'COCO', 'DET', 'YOUTUBEBB', 'MOT')

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = 'training_dataset/vid/crop511'
__C.DATASET.VID.ANNO = 'training_dataset/vid/train.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = 100000  # repeat until reach NUM_USE

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = 'training_dataset/yt_bb/crop511'
__C.DATASET.YOUTUBEBB.ANNO = 'training_dataset/yt_bb/train.json'
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = -1  # use all not repeat

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = 'training_dataset/coco/crop511'
__C.DATASET.COCO.ANNO = 'training_dataset/coco/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.NUM_USE = -1
#
__C.DATASET.DET = CN()
__C.DATASET.DET.ROOT = 'training_dataset/det/crop511'
__C.DATASET.DET.ANNO = 'training_dataset/det/train.json'
__C.DATASET.DET.FRAME_RANGE = 1
__C.DATASET.DET.NUM_USE = -1

__C.DATASET.VISDRONE = CN()
__C.DATASET.VISDRONE.ROOT = 'training_dataset/visdrone/crop511'
__C.DATASET.VISDRONE.ANNO = 'training_dataset/visdrone/train.json'
__C.DATASET.VISDRONE.FRAME_RANGE = 1
__C.DATASET.VISDRONE.NUM_USE = -1

__C.DATASET.LASOT = CN()
__C.DATASET.LASOT.ROOT = 'training_dataset/lasot/crop511'
__C.DATASET.LASOT.ANNO = 'training_dataset/lasot/train.json'
__C.DATASET.LASOT.FRAME_RANGE = 100
__C.DATASET.LASOT.NUM_USE = -1

__C.DATASET.VIDEOS_PER_EPOCH = 600000
# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'res50'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Pretrained all weights
__C.PRETRAINED = ''

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.LAYER = 1

__C.ADJUST.FUSE = 'avg'

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"

# ------------------------------------------------------------------------ #
# RPN options
# ------------------------------------------------------------------------ #
__C.RPN = CN()

# Whether to use rpn
__C.RPN.RPN = True

# RPN type
__C.RPN.TYPE = 'MultiRPN'

__C.RPN.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# mask options
# ------------------------------------------------------------------------ #
__C.MASK = CN()

# Whether to use mask generate segmentation
__C.MASK.MASK = False

# Mask type
__C.MASK.TYPE = "MaskCorr"

__C.MASK.KWARGS = CN(new_allowed=True)

__C.REFINE = CN()

# Mask refine
__C.REFINE.REFINE = False

# Refine type
__C.REFINE.TYPE = "Refine"

# ------------------------------------------------------------------------ #
# Anchor options
# ------------------------------------------------------------------------ #
__C.ANCHOR = CN()

# Anchor stride
__C.ANCHOR.STRIDE = 8

# Anchor ratios
__C.ANCHOR.RATIOS = [0.33, 0.5, 1, 2, 3]

# Anchor scales
__C.ANCHOR.SCALES = [8]

# Anchor number
__C.ANCHOR.ANCHOR_NUM = len(__C.ANCHOR.RATIOS) * len(__C.ANCHOR.SCALES)

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamRPNTracker'

# siamfc params
__C.TRACK.RESPONSE_UP_STRIDE = 16
__C.TRACK.TOTAL_STRIDE = 8
__C.TRACK.SCALE_NUM = 3
__C.TRACK.SCALE_STEP = 1.0375

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.3

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 303

# Base size
__C.TRACK.BASE_SIZE = 8

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

# Long term lost search size
__C.TRACK.LOST_INSTANCE_SIZE = 831

# Long term confidence low
__C.TRACK.CONFIDENCE_LOW = 0.85

# Long term confidence high
__C.TRACK.CONFIDENCE_HIGH = 0.998

# Mask threshold
__C.TRACK.MASK_THERSHOLD = 0.30

# Mask output size
__C.TRACK.MASK_OUTPUT_SIZE = 127

################################
# @cmd params
__C.TRACK.USE_CLASSIFIER = False
__C.TRACK.VISUALIZE_CLASS = False
__C.TRACK.DEBUG_CLASS = False
__C.TRACK.ANALYZE_CONVERGENCE = False
__C.TRACK.SEED = 12345  # [0, 1, 12, 123, 1234, 12345, 123456]
__C.TRACK.COEE_CLASS = 0.5

# @network params
#(projection matrix)
__C.TRACK.PROJECTION_REG = 1e-4
# first layer (compression)
__C.TRACK.USE_PROJECTION_MATRIX = True
__C.TRACK.UPDATE_PROJECTION_MATRIX = True
__C.TRACK.COMPRESSED_DIM = 64
__C.TRACK.PROJ_INIT_METHOD = 'randn'
__C.TRACK.PROJECTION_ACTIVATION = 'none'  # ('none', 'relu', 'elu' or 'mlu')
# second layer (attention)
__C.TRACK.USE_ATTENTION_LAYER = False
__C.TRACK.ATT_FC1_REG = 1e-4
__C.TRACK.ATT_FC2_REG = 1e-4
__C.TRACK.ATT_INIT_METHOD = 'randn'
__C.TRACK.ATT_ACTIVATION = 'relu'
__C.TRACK.SPATIAL_ATTENTION = 'none'  # [''pool']
__C.TRACK.CHANNEL_ATTENTION = False
# third layer (filter)
__C.TRACK.FILTER_REG = 1e-1
__C.TRACK.KERNEL_SIZE = (4, 4)
__C.TRACK.FILTER_INIT_METHOD = 'randn'  # method for initializing the filter
__C.TRACK.RESPONSE_ACTIVATION = ('mlu', 0.05)

# @augmentation params
__C.TRACK.USE_AUGMENTATION = True
__C.TRACK.AUGMENTATION_SHIFT = False
__C.TRACK.AUGMENTATION_SCALE = False
__C.TRACK.AUGMENTATION_FLIPLR = True
__C.TRACK.AUGMENTATION_ROTATE = [
    5, -5, 10, -10, 20, -20, 30, -30, 45, -45, -60, 60
]
__C.TRACK.AUGMENTATION_BLUR = [(2, 0.2), (0.2, 2), (3, 1), (1, 3), (2, 2)]
__C.TRACK.AUGMENTATION_RELATIVESHIFT = [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6),
                                        (-0.6, -0.6)]
__C.TRACK.AUGMENTATION_DROUPOUT = (7, 0.2)
__C.TRACK.AUGMENTATION_EXPANSION_FACTOR = 2
__C.TRACK.RANDOM_SHIFT_FACTOR = 1 / 3

# @optimization params
__C.TRACK.CG_OPTIMIZER = True
__C.TRACK.LEARNING_RATE = 0.01
__C.TRACK.INIT_SAMPLES_MINIMUM_WEIGHT = 0.25
__C.TRACK.SAMPLE_MEMORY_SIZE = 250
__C.TRACK.OUTPUT_SIGMA_FACTOR = 0.25

# GNCG - Gauss-Newton CG
__C.TRACK.OPTIMIZER = 'GaussNewtonCG'  # ['GaussNewtonCG', 'GradientDescentL2', 'NewtonCG', 'GradientDescent']
__C.TRACK.INIT_CG_ITER = 60
__C.TRACK.INIT_GN_ITER = 6
__C.TRACK.TRAIN_SKIPPING = 10
__C.TRACK.CG_ITER = 5
__C.TRACK.POST_INIT_CG_ITER = 0
__C.TRACK.FLETCHER_REEVES = False  # Use the Fletcher-Reeves (true) or Polak-Ribiere (false) formula in the Conjugate Gradient
__C.TRACK.STANDARD_ALPHA = True
__C.TRACK.CG_FORGETTING_RATE = False

# SGD - standard gradient descent
__C.TRACK.OPTIMIZER_STEP_LENGTH = 10
__C.TRACK.OPTIMIZER_MOMENTUM = 0.9

# ADVANCED_LOCALIZATION - hard negative mining & absence assessment
__C.TRACK.ADVANCED_LOCALIZATION = True
__C.TRACK.TARGET_NOT_FOUND_THRESHOLD = 0.25
__C.TRACK.TEMPLATE_UPDATE = False
__C.TRACK.TARGET_UPDATE_THRESHOLD = 0.75
__C.TRACK.TARGET_UPDATE_SKIPPING = 5
__C.TRACK.TAU_REGRESSION = 0.6
__C.TRACK.TAU_CLASSIFICATION = 0.5
__C.TRACK.DISTRACTOR_THRESHOLD = 0.8
__C.TRACK.HARD_NEGATIVE_THRESHOLD = 0.5
__C.TRACK.TARGET_NEIGHBORHOOD_SCALE = 2.2
__C.TRACK.DISPLACEMENT_SCALE = 0.8
__C.TRACK.HARD_NEGATIVE_LEARNING_RATE = 0.02
__C.TRACK.HARD_NEGATIVE_CG_ITER = 5
__C.TRACK.SHORT_TERM_DRIFT = False
