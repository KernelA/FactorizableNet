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
        self.path_opt = 'options/models/VRD.yaml'
        self.dir_logs = None
        self.model_name = None
        self.dataset_option = None
        self.batch_size = None
        self.workers = 1
        self.lr = None
        self.learning_rate = None
        self.epochs = None
        self.eval_epochs = 1
        self.print_freq = 1000
        self.step_size = None
        self.optimizer = None
        self.infinite = False
        self.iter_size = 1
        self.loss_weight = True
        self.clip_gradient = True
        self.MPS_iter = None
        self.dropout = None
        self.resume = None
        self.pretrained_model = 'output/trained_models/Model-VRD.h5'
        self.warm_iters = -1
        self.optimize_MPS = False
        self.start_epoch = 0
        self.save_all_from = None
        self.evaluate = True
        self.evaluate_object = False
        self.use_normal_anchors = False
        self.seed = 1
        self.rpn = None
        self.nms = -1
        self.triplet_nms = 0.4
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

if args.path_opt is not None:
    # with open(args.path_opt, 'r') as handle:
    handle = open(args.path_opt, 'r')
    options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
    handle.close()
    options = utils.update_values(options, options_yaml)
    # with open(options['data']['opts'], 'r') as f:
    f = open(options['data']['opts'], 'r')
    data_opts = yaml.load(f, Loader=yaml.FullLoader)
    options['data']['dataset_version'] = data_opts.get('dataset_version', None)
    options['opts'] = data_opts
    f.close()

print '## options'
pprint(options)

lr = options['optim']['lr']
options = get_model_name(options)
print 'Checkpoints are saved to: {}'.format(options['logs']['dir_logs'])

train_set = getattr(datasets, options['data']['dataset'])(data_opts, 'train',
                            dataset_option=options['data'].get('dataset_option', None),
                            batch_size=options['data']['batch_size'],
                            use_region=options['data'].get('use_region', False))

test_set = getattr(datasets, options['data']['dataset'])(data_opts, 'test',
                            dataset_option=options['data'].get('dataset_option', None),
                            batch_size=options['data']['batch_size'],
                            use_region=options['data'].get('use_region', False))


model = getattr(models, options['model']['arch'])(train_set, opts = options['model'])

# pass enough message for anchor target generation
train_set._feat_stride = model.rpn._feat_stride
train_set._rpn_opts = model.rpn.opts

for key in train_set[0].keys():
    print(key + ' : ' + str(type(train_set[0][key])))