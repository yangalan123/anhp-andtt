# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Test structured neural Hawkes process
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

from anhp.utils.log import LogWriter, LogReader
# from anhp.esm.manager import Manager
from andtt.esm.manager import Manager
from anhp.eval.sigtest import Bootstrapping

import argparse
__author__ = 'Hongyuan Mei, Chenghao Yang'

class Tester(Manager):

    def __init__(self, args):
        tic = time.time()
        data_splits = [(args.Split, args.Ratio)]
        Manager.__init__(self, args, data_splits )
        self.create_downsampler(
            args.Split, args.DownSampleMode, args.DownSampleSize)
        self.load_tdb([args.Split]) # load temporal db for seqs
        with open(args.PathModel+'_idx_and_dim.pkl', 'rb') as f: 
            idx_and_dim = pickle.load(f)
        params = torch.load(args.PathModel, map_location='cpu')
        self.load_params(idx_and_dim, params)
        self.update_params_given_tdb() # may update nothing because all are loaded
        self.create_thinningsampler(args.NumSample, args.NumExp)
        print(f"time spent on initializatin : {time.time()-tic:.2f}")

    #@profile
    def run(self):

        print(f"start testing on {self.args.Split} data of domain {self.args.Domain} ... ")
        bs = Bootstrapping()

        tic = time.time()
        # testing with loglik 
        print("testing on loglik")
        total_loglik, total_num_token, all_res_loglik, all_res_loglik_token = \
            self.get_logprob_one_epoch(self.args.Split, self.args.Split)
        avg_loglik = total_loglik / total_num_token
        # significant tests with loglik 
        print(f"loglik is {avg_loglik:.4f}")
        print(f"time is {(time.time() - tic):.2f}")
        print("bootstrapping for significant tests")
        bs.run(all_res_loglik)

        all_res = {
            'loglik': all_res_loglik, 
            'loglik_token': all_res_loglik_token, 
            'pred': None
        }

        if self.args.Predict: 
            tic = time.time()
            print("testing on prediction")
            # testing with prediction 
            # get prediction and ground truth 
            all_res_pred = self.get_pred_and_gold_one_epoch(
                self.args.Split, self.args.Split)
            # finish
            all_res['pred'] = all_res_pred
        with open(self.args.PathResult, 'wb') as f:
            pickle.dump(all_res, f)
        print("\ntesting finished")