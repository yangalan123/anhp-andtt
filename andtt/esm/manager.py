# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
manage structured neural Hawkes process
used by trainer and tester
@author: hongyuan
"""

import pickle
import time
import numpy
import random
import os
import datetime
from itertools import chain

import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
NOTE
need to carefully check if EventSampler needs to be modified in any way
"""
from anhp.data.NHPDataset import createDataLoader
from anhp.esm.thinning import EventSampler
# from anhp.logic.neuraldatalog import NeuralDatalog as Datalog

"""
NOTE
rename modules to match old code structure
"""
from andtt.logic.xfmr_nd import XFMRNeuralDatalog as Datalog
from andtt.esm.helpers import get_subseq
from andtt.esm.helpers import get_logprob_and_grads, get_logprob
from andtt.esm.helpers import get_logprob_and_grads_cp
from andtt.esm.helpers import get_prediction_and_gold

__author__ = 'Hongyuan Mei, Chenghao Yang'


class Manager(object):

    def __init__(self, args, data_splits):
        self.eps = numpy.finfo(float).eps
        self.args = args
        self.set_seed()
        self.set_device()
        self.create_database()
        self.load_data(data_splits)
        self.down_sampler = dict()
        self.thinning_sampler = None # will be filled by tester
        self.tdb_loaded = False # init as False 
        """
        create or load tdb should be declared in builder, trainer, tester, etc
        """

    def set_seed(self):
        random.seed(self.args.Seed)
        numpy.random.seed(self.args.Seed)
        torch.manual_seed(self.args.Seed)

    def set_device(self):
        device = 'cuda' if self.args.UseGPU else 'cpu'
        self.device = torch.device(device)

    def create_database(self):
        print("reading domain knowledge and data...")
        tic = time.time()
        # self.datalog = Datalog(self.args.LSTMPool, self.args.UpdateMode)
        self.datalog = Datalog(
            # te_embed_dim=self.args.TimeEmbeddingDim,
            # te_mode=self.args.TimeEmbeddingMode,
            # layer=self.args.Layer
            args=self.args
        )
        self.datalog.load_rules_from_database(
            os.path.join(self.args.PathDomain, f'{self.args.Database}.db'))
        self.datalog.init_params()
        if self.args.UseGPU:
            self.datalog.cuda()
        message = f"time to init datalog database is {time.time()-tic:.2f}"
        print(message)

    """
    NOTE
    here is tricky 
    it uses old method, but param forms are new (e.g., attention params)
    so there might be discrepancy
    """
    def load_params(self, idx_and_dim, params): 
        print(f"load previously saved params (and their idx and dim)")
        self.datalog.load_params(idx_and_dim, params)

    def load_data(self, data_splits): 
        print(f"load data with split specs : {data_splits}")
        self.data = createDataLoader()
        self.data.load_data(self.args.PathDomain, data_splits)

    def save_params(self): 
        #print(f"save params : {self.datalog.count_params()} in total")
        self.datalog.save_params(self.args.PathSave)
        if hasattr(self, "optimizer"):
            torch.save(self.optimizer.state_dict(), self.args.PathSave + "_optimizer.pt")

    def create_downsampler(self, name, mode, size): 
        """
        may downsample for train / dev / test
        """
        self.down_sampler[name] = DownSampler(mode, size)

    def create_thinningsampler(self, num_sample, num_exp): 
        self.thinning_sampler = EventSampler(num_sample, num_exp)
    
    """
    about temporal database
    """

    def create_tdb(self, data_splits): 
        print(f"create temporal database with split specs : {data_splits}")
        loc = os.path.join(self.args.PathDomain, 'tdbcache')
        for s in data_splits: 
            print(f"create tdb for {len(self.data.seq[s])} {s} seqs")
            tic = time.time()
            self.datalog.create_tdb( 
                self.args.Database, s, self.data.seq[s], loc, self.args.TrackPeriod )
            print(f"it takes {time.time() - tic:.2f} seconds")

    def load_tdb(self, data_splits): 
        print(f"load temporal database with split specs : {data_splits}")
        loc = os.path.join(self.args.PathDomain, 'tdbcache')
        for s in data_splits: 
            num = len(self.data.seq[s])
            print(f"load tdb for {num} {s} seqs")
            tic = time.time()
            self.datalog.load_tdb(
                self.args.Database, s, num, loc )
            print(f"it takes {time.time() - tic:.2f} seconds")
        self.tdb_loaded = True # set flag to True

    def update_params_given_tdb(self): 
        print(f"update params after loading temporal database")
        assert self.tdb_loaded, "can't update params until tdb loaded"
        tic = time.time()
        self.datalog.update_params_given_tdb()
        message = f"time to update params is {time.time()-tic:.2f}"
        print(message)

    """
    about log prob
    """
    """
    NOTE 
    these methods are identical to those in snhp.esm.manager 
    so are other methods in this class 
    why not inherent that class?
    """

    def get_logprob_one_epoch(self, split, sampler_name):
        total_loglik, total_num_token = 0.0, 0.0
        all_res = [] # seq-level
        all_res_token = [] # token-level 
        for seq_i in range(self.data.sizes[split]):
            loglik, num_token, all_loglik = self.get_logprob_one_seq(seq_i, split, sampler_name)
            total_loglik += loglik
            total_num_token += num_token
            all_res.append( (loglik, num_token) )
            all_res_token += [ (x, 1.0) for x in all_loglik ]
        return total_loglik, total_num_token, all_res, all_res_token

    def get_logprob_one_seq(self, seq_id, split, sampler_name): 
        arguments = self.get_arguments_for_logprob(seq_id, split, sampler_name)
        loglik, num_token, all_loglik = get_logprob(arguments)
        return loglik, num_token, all_loglik

    def accum_grads_one_seq(self, seq_id, split, sampler_name): 
        arguments = self.get_arguments_for_logprob(seq_id, split, sampler_name)
        log_prob, num_event, num_inten = get_logprob_and_grads(arguments)
        # num_inten : # of intensities computed for this log prob
        return log_prob, num_event, num_inten

    def get_arguments_for_logprob(self, seq_id, split, sampler_name):
        # should match get_logprob_with_graph in helpers
        if 'train' in split: 
            multiplier = self.args.Multiplier
        else: 
            multiplier = self.args.DevMultiplier

        arguments = [
            seq_id, split, 
            get_subseq(self.data, split, seq_id),
            self.data.mask[split][seq_id], 
            self.datalog, 
            self.down_sampler[sampler_name], 
            multiplier, self.eps, self.device,
            self.args.MemorySize
        ]
        return arguments

    def accum_grads_one_seq_cp(self, seq_id, split, sampler_name): 
        arguments = self.get_arguments_for_logprob(seq_id, split, sampler_name)
        log_prob, num_event, num_inten = get_logprob_and_grads_cp(
            arguments, 
            (self.args.CheckPoint, self.optimizer, self.args.BatchSize)
        )
        # num_inten : # of intensities computed for this log prob
        return log_prob, num_event, num_inten

    """
    NOTE
    Dec 27, 2020
    already checked log-prob methods, they are good 
    only question is: why not inherent old Manager class since methods are mostly identical
    next step: prediction
    """

    """
    about prediction
    """
    """
    NOTE
    prediction-related code not changed yet
    """

    def get_pred_and_gold_one_epoch(self, split, sampler_name): 
        all_res = []
        for seq_i in range(self.data.sizes[split]): 
            if (seq_i + 1) % self.args.TrackPeriod == 0: 
                print(f"predicting for seq i : {seq_i}")
            res_seq_i = self.get_pred_and_gold_one_seq(
                seq_i, split, sampler_name)
            all_res.append( res_seq_i )
        return all_res

    def get_pred_and_gold_one_seq(self, seq_id, split, sampler_name): 
        # to compute prediction for this sequence
        arguments = self.get_arguments_for_pred(seq_id, split, sampler_name)
        rst_seq = get_prediction_and_gold(arguments)
        return rst_seq

    def get_arguments_for_pred(self, seq_id, split, sampler_name): 
        assert self.thinning_sampler is not None, "how to predict without thinning?!"
        arguments = [
            seq_id, split, 
            get_subseq(self.data, split, seq_id), 
            self.data.mask[split][seq_id], 
            self.datalog, 
            self.down_sampler[sampler_name], 
            self.thinning_sampler, 
            self.args.NumObject, # # of objects (including sub if have to) to predict
            self.eps, self.device, self.args.Verbose,
            self.args.MemorySize
        ]
        return arguments
