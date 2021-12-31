# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Build temporal databases for neural Datalog through time (NDTT)
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

from andtt.esm.builder import Builder

import argparse
__author__ = 'Hongyuan Mei'


def main():

    parser = argparse.ArgumentParser(
        description='Build temporal databases for Transformer version of neural Datalog through time (T-NDTT)')

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
        '-s', '--Split', required=True, type=str, help='what split to use?',
        choices = ['train', 'dev', 'test']
    )
    parser.add_argument(
        '-r', '--Ratio', default=1.0, type=float, help='fraction of data to use'
    )
    parser.add_argument(
        '-tp', '--TrackPeriod', default=1, type=int, help='# seqs each print for tdb creation'
    )
    parser.add_argument(
        '-gpu', '--UseGPU', action='store_true', help='use GPU?'
    )
    parser.add_argument(
        '-sd', '--Seed', default=12345, type=int, help='random seed'
    )

    """
    NOTE 
    we even need to specify dim(te) for building! 
    we should really find a better design
    """
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
    just like LSTMPool or UpdateMode, it is just dummy, right? 
    it actully only matters when the params are being created
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

    args = parser.parse_args()
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()

    args.Version = torch.__version__
    args.ID = id_process
    args.TIME = time_current

    path_storage = os.path.abspath(args.PathStorage)
    args.PathDomain = os.path.join(path_storage, 'domains', args.Domain)

    """
    dummy values: they don't affect building temporal databases
    but they have to be specified to create neural datalog interface
    """
    args.LSTMPool = 'full'
    args.UpdateMode = 'sync'

    builder = Builder(args)
    builder.run()

if __name__ == "__main__": main()
