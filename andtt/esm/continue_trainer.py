# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Train structured neural Hawkes process
using maximum likelihod estimation (MLE)
from pre-saved models

@author: hongyuan
"""

import pickle
import time
import numpy
import random
import os
import datetime

import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ndtt.io.log import LogWriter, LogReader
# from ndtt.esm.trainer import Manager, Trainer
from andtt.esm.trainer import Manager, Trainer

import argparse
__author__ = 'Hongyuan Mei, Chenghao Yang'


class ContinueTrainer(Trainer):

    def __init__(self, args):
        tic = time.time()
        data_splits = [('train', args.TrainRatio), ('dev', args.DevRatio)]
        Manager.__init__(self, args, data_splits)
        self.create_downsampler(
            'train', args.TrainDownSampleMode, args.TrainDownSampleSize)
        self.create_downsampler(
            'dev', args.DevDownSampleMode, args.DevDownSampleSize)
        self.load_tdb(['train', 'dev']) # load temporal db for seqs
        """
        MUST collect all params given temporal database before training starts!
        s.t. optimizer gets all of them
        otherwise, if we generate params on the fly of training 
        some of them may not be accessed by optimizer, thus not trained
        """
        with open(args.PathSave+'_idx_and_dim.pkl', 'rb') as f: 
            idx_and_dim = pickle.load(f)
        params = torch.load(args.PathSave, map_location='cpu')
        self.load_params(idx_and_dim, params)
        self.update_params_given_tdb() 
        # update params given tdb
        # may update nothing because all are loaded
        """
        track # of params before training 
        """
        self.args.NumParams = self.datalog.count_params()
        self.optimizer = optim.Adam(
            self.datalog.chain_params(), lr=args.LearnRate
        )
        if os.path.exists(args.PathSave + "_optimizer.pt"):
            self.optimizer.load_state_dict(torch.load(args.PathSave + "_optimizer.pt"))
        self.optimizer.zero_grad() # init clear
        self.log = LogWriter(args.PathLog, vars(self.args) )
        # use self.args not args, cuz init function may add things to args
        self.log.initBest()
        self.max_episode = args.MaxEpoch * self.data.sizes['train']
        print(f"time spent on initializatin : {time.time()-tic:.2f}")
    