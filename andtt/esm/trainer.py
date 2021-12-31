# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Train Transformer structured neural Hawkes process
using maximum likelihod estimation (MLE)

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

from snhp.io.log import LogWriter
# from snhp.esm.manager import Manager
from andtt.esm.manager import Manager

import argparse
__author__ = 'Hongyuan Mei, Chenghao Yang'


class Trainer(Manager):

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
        self.update_params_given_tdb() # update params given tdb
        """
        track # of params before training 
        """
        self.args.NumParams = self.datalog.count_params()
        self.optimizer = optim.Adam(
            self.datalog.chain_params(), lr=args.LearnRate,
            weight_decay=args.WeightDecay if hasattr(args, "WeightDecay") else 0
        )
        self.optimizer.zero_grad() # init clear
        self.log = LogWriter(args.PathLog, vars(self.args) )
        # use self.args not args, cuz init function may add things to args
        self.log.initBest()
        self.max_episode = args.MaxEpoch * self.data.sizes['train']
        print(f"time spent on initializatin : {time.time()-tic:.2f}")

    """
    NOTE
    do we use -1 or other? 
    do we actually call run_entire_seq or run_check_point? 
    i expect to use check_point but it is okay to start with entire_seq
    """
    def run(self): 
        if self.args.CheckPoint == -1 : 
            self.run_entire_seq()
        elif self.args.CheckPoint > 0 : 
            """
            for seq that is too long, step() after entire seq is too infrequent
            need to update gradients at each checkpoint inside each seq
            """
            self.run_check_point()
        else: 
            raise Exception(f"Unknown check point # : {self.args.CheckPoint}")

    #@profile
    def run_entire_seq(self):
        print("start training ... ")
        time_train = 0.0
        for episode in range(self.max_episode):
            tic = time.time()
            seq_id = int(episode % self.data.sizes['train'])
            log_lik, num_token, _ = self.accum_grads_one_seq(seq_id, 'train', 'train')
            #self.datalog.check_grads()
            if (episode+1) % self.args.BatchSize == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                #self.datalog.check_params()
            time_train += (time.time() - tic)

            if episode % self.args.TrackPeriod == self.args.TrackPeriod - 1:
                message = f"time to train {self.args.TrackPeriod} episodes is {time_train:.2f}," \
                          f"train log-lik is {log_lik / num_token}"
                self.log.checkpoint(message)
                print(message)
                self.validate(episode)
                time_train = 0.0

        message = "training finished"
        self.log.checkpoint(message)
        print(message)


    def run_check_point(self): 
        print(f"start training ... ")
        time_train = 0.0
        for episode in range(self.max_episode):
            tic = time.time()
            seq_id = int(episode % self.data.sizes['train'])
            log_lik, num_token, _ = self.accum_grads_one_seq_cp(seq_id, 'train', 'train')
            time_train += (time.time() - tic)

            if episode % self.args.TrackPeriod == self.args.TrackPeriod - 1: 
                message = f"time to train {self.args.TrackPeriod} episodes is {time_train:.2f}," \
                          f"training log-lik is {log_lik / num_token}"
                self.log.checkpoint(message)
                print(message)
                self.validate(episode)
                time_train = 0.0 

        message = "training finished"
        self.log.checkpoint(message)
        print(message)

    
    def validate(self, episode):
        epoch_id = int(episode / self.data.sizes['train'])
        seq_id = int(episode % self.data.sizes['train'])
        message = "validating at episode-{} (seq-{} of training epoch-{})".format(
            episode, seq_id, epoch_id )
        print(message)
        tic = time.time()
        # validate on dev data
        total_loglik, total_num_token, _, _ = self.get_logprob_one_epoch('dev', 'dev')
        avg_loglik = total_loglik / total_num_token
        # finish
        message = "loglik is {:.4f} after episode-{} (seq-{} of epoch-{})".format(
            avg_loglik, episode, seq_id, epoch_id)
        self.log.checkpoint(message)
        print(message)
        updated = self.log.updateBest('loglik', avg_loglik, episode)
        message = "current best loglik is {:.4f} (updated at episode-{})".format(
            self.log.current_best['loglik'], self.log.episode_best)
        if updated:
            message += f", best updated at this episode"
            self.save_params()
            #message += f", save params : {self.datalog.count_params()} in total"
        self.log.checkpoint(message)
        print(message)
        message = f"time for dev is {(time.time() - tic):.2f}"
        self.log.checkpoint(message)
        print(message)
