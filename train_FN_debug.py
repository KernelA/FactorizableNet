import os
import os.path as osp
import shutil
import time
import random
import numpy as np
import numpy.random as npr
import argparse
import yaml
import click
from pprint import pprint
# To restore the testing results for further analysis
import cPickle
import sys
sys.stdout.flush()


import torch

from lib import network
from lib.utils.timer import Timer
import lib.datasets as datasets
from lib.utils.FN_utils import get_model_name, group_features, get_optimizer
import lib.utils.general_utils as utils
import lib.utils.logger as logger
import models
from models.HDN_v2.utils import save_checkpoint, load_checkpoint, save_results, save_detections

from models.modules.dataParallel import DataParallel


import pdb

args = {
    'path_opt' : 'options/FN_v4/map_v2.yaml',
    'dir_logs' : None,
    'model_name' : None, 
    'dataset_option' : None, 
    'workers' : 1, 
    'lr' : None, 
    'epochs' : None, 
    'eval_epochs' : 1, 
    'print_freq' : 1000, 
    'step_size' : None, 
    'optimizer' : None, 
    'infinite' : False, 
    'iter_size' : 1, 
    'loss_weight' : True, 
    'clip_gradient' : True, 
    'MPS_iter' : None, 
    'dropout' : None, 
    'resume' : None, 
    'pretrained_model' : None, 
    'warm_iters' : -1, 
    'optimize_MPS' : False, 
    'start_epoch' : 0, 
    'save_all_from' : None, 
    'evaluate' : False, 
    'evaluate_object' : False, 
    'use_normal_anchors' : False, 
    'seed' : 1, 
    'rpn' : None, 
    'nms' : -1, 
    'triplet_nms' : 0.4, 
    'use_gt_boxes' : False
}

overall_train_loss = network.AverageMeter()
overall_train_rpn_loss = network.AverageMeter()
overall_gradients_norm_logger = network.LoggerMeter()

is_best = False
best_recall = [0., 0.]
best_recall_phrase = [0., 0.]
best_recall_pred = [0., 0.]