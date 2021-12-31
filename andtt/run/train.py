# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Train neural Datalog through time (NDTT)
using maximum likelihood estimation (MLE)

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

from andtt.esm.trainer import Trainer

import argparse
__author__ = 'Hongyuan Mei, Chenghao Yang'


def get_args(method='MLE'): 

    parser = argparse.ArgumentParser(description=f'training neural Datalog through time (NDTT) using {method}')

    parser.add_argument(
        '-d', '--Domain', required=True, type=str, help='which domain to work on?'
    )
    parser.add_argument(
        '-db', '--Database', required=True, type=str, help='which database to use?'
    )
    parser.add_argument(
        '-ps', '--PathStorage', type=str, default='../..',
        help='Path of storage which stores domains (with data), logs, results, etc. \
            Must be local (e.g. no HDFS allowed)'
    )
    parser.add_argument(
        '-cp', '--CheckPoint', default=-1, type=int, 
        help='every # tokens (>1) in a seq to accumulate gradients (and cut compute graph), -1 meaning entire seq'
    )
    parser.add_argument(
        '-bs', '--BatchSize', default=1, type=int, 
        help='# checkpoints / seqs to update parameters'
    )
    parser.add_argument(
        '-tp', '--TrackPeriod', default=1000, type=int, help='# seqs to train for each dev'
    )
    parser.add_argument(
        '-m', '--Multiplier', default=1, type=float,
        help='constant of N=O(I), where N is # of sampled time points for integral'
    )
    parser.add_argument(
        '-dm', '--DevMultiplier', default=1, type=int,
        help='constant of N=O(I), where N is # of sampled time points for integral'
    )
    parser.add_argument(
        '-tr', '--TrainRatio', default=1.0, type=float, help='fraction of training data to use'
    )
    parser.add_argument(
        '-dr', '--DevRatio', default=1.0, type=float, help='fraction of dev data to use'
    )
    parser.add_argument(
        '-me', '--MaxEpoch', default=20, type=int, help='max # training epochs'
    )
    parser.add_argument(
        '-lr', '--LearnRate', default=1e-3, type=float, help='learning rate'
    )
    parser.add_argument(
        '-wd', '--WeightDecay', default=0, type=float, help='weight decay'
    )
    parser.add_argument(
        '-np', '--NumProcess', default=1, type=int, help='# of processes used, default is 1'
    )
    parser.add_argument(
        '-nt', '--NumThread', default=1, type=int, help='OMP NUM THREADS'
    )
    parser.add_argument(
        '-tdsm', '--TrainDownSampleMode', default='none', type=str, choices=['none', 'uniform'], 
        help='for training, how do you want to down sample it? none? uniform?'
    )
    parser.add_argument(
        '-tdss', '--TrainDownSampleSize', default=1, type=int, 
        help='for training, down sample size, 1 <= dss <= K'
    )
    parser.add_argument(
        '-ddsm', '--DevDownSampleMode', default='none', type=str, choices=['none', 'uniform'], 
        help='for dev, how do you want to down sample it? none? uniform?'
    )
    parser.add_argument(
        '-ddss', '--DevDownSampleSize', default=1, type=int, 
        help='for dev, down sample size, 1 <= dss <= K'
    )
    parser.add_argument(
        '-teDim', '--TimeEmbeddingDim', default=100, type=int,
        help='the dimensionality of time embedding'
    )
    parser.add_argument(
        '-layer', '--Layer', default=3, type=int,
        help='the number of layers of Transformer'
    )
    parser.add_argument(
        '-attemp', '--AttentionTemperature', default=1.0, type=float, help='temperature of softmax used in attention'
    )
    """
    NOTE
    it is really bad that users have to specify time embedding
    we should design how time embedding dim is automatically determined
    """
    parser.add_argument(
        '-tem', '--TimeEmbeddingMode', default='Sine', type=str, choices=["Sine", "Linear"],
        help='how do you want to get time embedding?'
    )
    parser.add_argument(
        '-intenmode', '--IntensityComputationMode', default='extra_layer', type=str, choices=["extra_dim", "extra_layer"],
        help='how do you want to compute the intensities? via extra_layer or extra_dim?'
    )
    parser.add_argument(
        '-mem', '--MemorySize', default=50, type=int,
        help='the number of past events that should be attended to in TransformerCell'
    )
    """
    NOTE
    LSTMPool and UpdateMode should all be removed
    NO, args can be removed, but values have to be specified, to make Manager
    for how to specify values, refer to build.py
    """
    parser.add_argument(
        '-lp', '--LSTMPool', default='full', type=str, choices=['full', 'simp'],
        help='for LSTM pooling, full(default):full-verison-in-paper;simp:a-simplification'
    )
    parser.add_argument(
        '-um', '--UpdateMode', default='sync', type=str, choices=['sync', 'async'],
        help='way of updating lstm after computed new cells'
    )
    parser.add_argument(
        '-gpu', '--UseGPU', action='store_true', help='use GPU?'
    )
    parser.add_argument(
        '-sd', '--Seed', default=12345, type=int, help='random seed'
    )
    args = parser.parse_args()
    return args

def aug_args_with_log(args): 
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()

    args.Version = torch.__version__
    args.ID = id_process
    args.TIME = time_current

    path_storage = os.path.abspath(args.PathStorage)
    args.PathDomain = os.path.join(path_storage, 'domains', args.Domain)

    dict_args = vars(args)
    folder_name = get_foldername(dict_args)

    path_logs = os.path.join(args.PathDomain, 'Logs', folder_name)
    os.makedirs(path_logs)
    args.PathLog = os.path.join(path_logs, 'log.txt')
    args.PathSave = os.path.join(path_logs, 'saved_model')

    args.NumProcess = 1
    #if args.MultiProcessing: 
    #    if args.NumProcess < 1:
    #        args.NumProcess = os.cpu_count()
    
    if args.NumThread < 1: 
        args.NumThread = 1
    
    print(f"mp num threads in torch : {torch.get_num_threads()}")
    if torch.get_num_threads() != args.NumThread: 
        print(f"not equal to NumThread arg ({args.NumThread})")
        torch.set_num_threads(args.NumThread)
        print(f"set to {args.NumThread}")
        assert torch.get_num_threads() == args.NumThread, "not set yet?!"

def get_foldername(dict_args): 
    """
    NOTE 
    adjust foler name attributes
    """
    args_used_in_name = [
        ['Database', 'db'],
        ['CheckPoint', 'cp'], 
        ['TrainRatio', 'tr'], 
        ['Multiplier', 'm'], 
        ['TrainDownSampleMode', 'tdsm'],
        ['TrainDownSampleSize', 'tdss'],
        ['DevDownSampleMode', 'ddsm'],
        ['DevDownSampleSize', 'ddss'],
        ['LSTMPool', 'lp'], 
        ['UpdateMode', 'um'],
        ["TimeEmbeddingDim", "teDim"],
        ["TimeEmbeddingMode", "tem"],
        ["Layer", "layer"],
        ["IntensityComputationMode", "intenMode"],
        ["AttentionTemperature", "attemp"],
        ["MemorySize", "mem"],
        ['LearnRate', 'lr'],
        ['WeightDecay', 'wd'],
        ['Seed', 'seed'],
        ['ID', 'id'],
    ]
    folder_name = list()
    for arg_name, rename in args_used_in_name:
        folder_name.append('{}-{}'.format(rename, dict_args[arg_name]))
    folder_name = '_'.join(folder_name)
    return folder_name

def main():

    args = get_args()
    aug_args_with_log(args)
    trainer = Trainer(args)
    trainer.run()

if __name__ == "__main__": main()
