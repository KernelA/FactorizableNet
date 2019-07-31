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

class arg():
    def __init__(self):
        self.path_opt = 'options/FN_v4/map_v2.yaml',
        self.dir_logs = None,
        self.model_name = None, 
        self.dataset_option = None, 
        self.batch_size = None, 
        self.workers = 1, 
        self.lr = None, 
        self.learning_rate = None, 
        self.epochs = None, 
        self.eval_epochs = 1, 
        self.print_freq = 1000, 
        self.step_size = None, 
        self.optimizer = None, 
        self.infinite = False, 
        self.iter_size = 1, 
        self.loss_weight = True, 
        self.clip_gradient = True, 
        self.MPS_iter = None, 
        self.dropout = None, 
        self.resume = None, 
        self.pretrained_model = None, 
        self.warm_iters = -1, 
        self.optimize_MPS = False, 
        self.start_epoch = 0, 
        self.save_all_from = None, 
        self.evaluate = False, 
        self.evaluate_object = False, 
        self.use_normal_anchors = False, 
        self.seed = 1, 
        self.rpn = None, 
        self.nms = -1, 
        self.triplet_nms = 0.4, 
        self.use_gt_boxes = False

args = arg()

overall_train_loss = network.AverageMeter()
overall_train_rpn_loss = network.AverageMeter()
overall_gradients_norm_logger = network.LoggerMeter()

is_best = False
best_recall = [0., 0.]
best_recall_phrase = [0., 0.]
best_recall_pred = [0., 0.]

options = {
    'logs': {
        'model_name': args.model_name,
        'dir_logs': args.dir_logs,
    },
    'data':{
        'dataset_option': args.dataset_option,
        'batch_size': args.batch_size,
    },
    'optim': {
        'lr': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr_decay_epoch': args.step_size,
        'optimizer': args.optimizer,
        'clip_gradient': args.clip_gradient,
    },
    'model':{
        'MPS_iter': args.MPS_iter,
        'dropout': args.dropout,
        'use_loss_weight': args.loss_weight,
    },
}



print '## args'
pprint(vars(args))
print '## options'
pprint(options)