# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Train neural Datalog through time (NDTT)
using maximum likelihood estimation (MLE)
from pre-saved models

@author: hongyuan
"""

import pickle
from shutil import copyfile
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

from andtt.esm.continue_trainer import ContinueTrainer
from snhp.io.log import LogReader

import argparse
__author__ = 'Hongyuan Mei'


def get_args(): 

    parser = argparse.ArgumentParser(description='training neural Datalog through time (NDTT) using MLE')

    parser.add_argument(
        '-d', '--Domain', required=True, type=str, help='which domain to work on?'
    )
    parser.add_argument(
        '-fn', '--FolderName', required=True, type=str, 
        help='base name of the folder to store the model (and log)?'
    )
    parser.add_argument(
        '-ps', '--PathStorage', type=str, default='../..',
        help='Path of storage which stores domains (with data), logs, results, etc. \
            Must be local (e.g. no HDFS allowed)'
    )
    parser.add_argument(
        '-me', '--MaxEpoch', default=10, type=int, help='max # continual training epochs'
    )
    parser.add_argument(
        '-lr', '--LearnRate', default=1e-3, type=float, help='learning rate'
    )
    parser.add_argument(
        '-np', '--NumProcess', default=1, type=int, help='# of processes used, default is 1'
    )
    parser.add_argument(
        '-nt', '--NumThread', default=1, type=int, help='OMP NUM THREADS'
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

    path_logs = os.path.join(args.PathDomain, 'Logs', args.FolderName)
    assert os.path.exists(path_logs)
    args.PathLog = os.path.join(path_logs, 'log.txt')
    log = LogReader(args.PathLog)
    saved_args = log.getArgs()
    args.PathSave = os.path.join(path_logs, 'saved_model')
    """
    back up old log and saved model
    """
    copyfile(
        args.PathLog, os.path.join(path_logs, 'log_bak.txt')
    )
    copyfile(
        args.PathSave, os.path.join(path_logs, 'saved_model_bak')
    )
    copyfile(
        args.PathSave + '_idx_and_dim.pkl', 
        args.PathSave + '_idx_and_dim_bak.pkl'
    )
    """
    grab saved arguments
    """
    args.Database = saved_args['Database']
    args.CheckPoint = saved_args['CheckPoint']
    args.BatchSize = saved_args['BatchSize']
    args.TrackPeriod = saved_args['TrackPeriod']
    args.Multiplier = saved_args['Multiplier']
    args.DevMultiplier = saved_args['DevMultiplier']
    args.TrainRatio = saved_args['TrainRatio']
    args.DevRatio = saved_args['DevRatio']
    args.TrainDownSampleMode = saved_args['TrainDownSampleMode']
    args.TrainDownSampleSize = saved_args['TrainDownSampleSize']
    args.DevDownSampleMode = saved_args['DevDownSampleMode']
    args.DevDownSampleSize = saved_args['DevDownSampleSize']
    args.LSTMPool = saved_args['LSTMPool']
    args.UpdateMode = saved_args['UpdateMode']
    args.TimeEmbeddingDim = saved_args["TimeEmbeddingDim"]
    args.TimeEmbeddingMode = saved_args["TimeEmbeddingMode"]
    args.UseGPU = saved_args['UseGPU']
    args.Seed = saved_args['Seed']
    args.Layer = saved_args['Layer']
    args.MemorySize = saved_args["MemorySize"]
    if "IntensityComputationMode" in saved_args:
        args.IntensityComputationMode = saved_args['IntensityComputationMode']
    if 'AttentionTemperature' in saved_args:
        args.AttentionTemperature = saved_args['AttentionTemperature']
    if "WeightDecay" in saved_args:
        args.WeightDecay = saved_args["WeightDecay"]
    else:
        args.WeightDecay = 0

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

def main():

    args = get_args()
    aug_args_with_log(args)
    trainer = ContinueTrainer(args)
    trainer.run()

if __name__ == "__main__": main()
