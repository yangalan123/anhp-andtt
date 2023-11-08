# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Build temporal databases for structured neural Hawkes process
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

from ndtt.io.log import LogWriter
from andtt.esm.manager import Manager

import argparse
__author__ = 'Hongyuan Mei, Chenghao Yang'


class Builder(Manager):

    def __init__(self, args):
        tic = time.time()
        Manager.__init__(
            self, args, [(args.Split, args.Ratio)])
        print(f"time spent on initializatin : {time.time()-tic:.2f}")

    def run(self): 
        self.create_tdb( [ self.args.Split ] )