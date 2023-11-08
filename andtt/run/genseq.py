# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Generate seqs from a neural Datalog through time (NDTT)

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

from dsl.base import DomainBase as Domain

from ndtt.models.cells import CTLSTMCell
from ndtt.models.event import Event
from ndtt.models.partition import NodePartition
from ndtt.models.manager import Manager
from ndtt.models.sampler import Sampler

from ndtt.functions.utils import getAbsPath

import argparse
__author__ = 'Hongyuan Mei'


class Generator(Manager):

    def __init__(self, args):
        tic = time.time()
        Manager.__init__(self, args, [], False)
        self.sampler = Sampler(num_sample=args.NumSample, num_exp=args.NumExp, device=self.device)
        print(f"time spent on initialization : {time.time()-tic:.2f}")

    #@profile
    def run(self):

        print(f"start sampling data ...")

        self.draw_one_set(self.args.NumTrain, self.args.NumEvent, os.path.join(self.args.PathDomain, 'train.pkl') )
        self.draw_one_set(self.args.NumDev, self.args.NumEvent, os.path.join(self.args.PathDomain, 'dev.pkl') )
        self.draw_one_set(self.args.NumTest, self.args.NumEvent, os.path.join(self.args.PathDomain, 'test.pkl') )
        self.draw_one_set(self.args.NumTest, self.args.NumEvent, os.path.join(self.args.PathDomain, 'test1.pkl') )
        self.draw_one_set(self.args.NumTest, self.args.NumEvent, os.path.join(self.args.PathDomain, 'test2.pkl') )

        print("saving model to {}".format(self.args.PathSave))
        self.save_model()

        message = "finish generating"
        print(message)


    def draw_one_set(self, num_seq, num_event, path_save):

        message = "generating {} seqs for {}".format(num_seq, os.path.basename(path_save) )
        print(message)

        tic = time.time()
        res = []

        for i in range(num_seq):

            if (i+1) % 10 == 0:
                print("{}-th seq".format(i+1))

            res.append(self.draw_one_seq(num_event) )

        message = f"time for generating is {(time.time() - tic):.2f}"
        print(message)

        if num_seq >= 5:
            message = f"the first 5 seqs are : "
            print(message)
            for i in range(5):
                self.print_one_seq(res[i])

        message = "saving seqs to {}".format(path_save)
        print(message)

        with open(path_save, 'wb') as f:
            pickle.dump(res, f)

        message = "finished for {}".format(os.path.basename(path_save) )
        print(message)

    def print_one_seq(self, seq):
        res = ''
        for event in seq:
            res += ' {} at {:.2f} ,'.format(event['name'], event['time'])
        print(res[:-1])

    #@profile
    def draw_one_seq(self, num_event):
        """
        reset partition values is necessary in this function 
        cuz the original pars are updated in this function
        """
        for p_name, p_obj in self.pars.items(): 
            p_obj.reset()

        time_last_event = 0.0
        num_token = 0
        seq = [{'name': 'BOS', 'time': 0.0}]

        while num_token < num_event + 1 :

            self.update_partitions(self.copy_add_dep(seq[-1]) )

            event = self.draw_one_event(time_last_event)
            seq.append(event)

            time_last_event = event['time']
            num_token += 1

        seq.append({'name': 'EOS', 'time': time_last_event+self.eps})

        return seq

    def copy_add_dep(self, event):
        """
        do something to get what this event affects 
        such info should have been pre-computed 
        """
        res = {}
        for k, v in event.items(): 
            res[k] = v 
        self.domain.getDependencyPerToken(res)
        return res 
    
    def draw_one_event(self, time_last_event):
        """
        we draw one event using thinning algorithm
        sizes : 
        accepted_times : 1 
        intensities_at_accepted_times : K * 1
        """
        accepted_times, _, intensities_at_accepted_times = self.sampler.draw_times_via_thinning(
            time_last_event, 
            self.domain, self.pars, 
            self.comps_out, self.events, self.constants, self.event2id, self.event_type_cnt, 
            # feed in self.event_type_cnt cuz we do not down sample event types 
            # when we draw events --- sum of intensities need to be exact 
            self.args.Adaptive, self.mpmgr)
        
        probs = intensities_at_accepted_times.squeeze(1)
        assert probs.dim() == 1, "probs not a vector?"
        assert accepted_times.dim() == 1, "probs not a vector?"
        assert accepted_times.size(0) == 1, "more than 1 accepted time?"
        probs /= probs.sum()
        event_name = self.id2event[int(torch.multinomial(probs, 1, True))]

        return {'name': event_name, 'time': float(accepted_times[0])}


def main():

    parser = argparse.ArgumentParser(description='generating seqs from neural Datalog through time (NDTT)')

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
        '-nexp', '--NumExp', default=100, type=int, 
        help='number of i.i.d. Exp(intensity_bound) draws at one time in thinning algorithm'
    )
    parser.add_argument(
        '-ne', '--NumEvent', default=20, type=int, help='# events per seq'
    )
    parser.add_argument(
        '-ntrain', '--NumTrain', default=1, type=int, help='# seqs for training'
    )
    parser.add_argument(
        '-ndev', '--NumDev', default=1, type=int, help='# seqs for dev'
    )
    parser.add_argument(
        '-ntest', '--NumTest', default=1, type=int, help='# seqs for test'
    )
    parser.add_argument(
        '-a', '--Adaptive', action='store_true', help='use adaptive thinning algorithm?'
    )
    parser.add_argument(
        '-f', '--Fractional', action='store_true', help='use fractional thinning algorithm?'
    )
    parser.add_argument(
        '-mp', '--MultiProcessing', action='store_true', help='use multiprocessing for computation?'
    )
    parser.add_argument(
        '-np', '--NumProcess', default=-1, type=int, help='# of processes used, default is same with # processors'
    )
    parser.add_argument(
        '-nt', '--NumThread', default=1, type=int, help='OMP NUM THREADS'
    )
    parser.add_argument(
        '-gpu', '--UseGPU', action='store_true', help='use GPU?'
    )
    parser.add_argument(
        '-sd', '--Seed', default=12345, type=int, help='random seed'
    )

    args = parser.parse_args()
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()

    args.Version = torch.__version__
    args.ID = id_process
    args.TIME = time_current

    path_storage = getAbsPath(args.PathStorage)
    args.PathDomain = os.path.join(path_storage, 'domains', args.Domain)
    args.PathSave = os.path.join(args.PathDomain, 'gen_model')

    args.NumSample = 1 # one i.i.d. draw for each instance of past history 

    if args.MultiProcessing: 
        raise Exception("MP Not Implemented for Sampling")
    else: 
        args.NumProcess = 1
    
    if args.NumThread < 1: 
        args.NumThread = 1

    print(f"mp num threads in torch : {torch.get_num_threads()}")
    if torch.get_num_threads() != args.NumThread: 
        print(f"not equal to NumThread arg ({args.NumThread})")
        torch.set_num_threads(args.NumThread)
        print(f"set to {args.NumThread}")
        assert torch.get_num_threads() == args.NumThread, "not set yet?!"

    gen = Generator(args)
    gen.run()


if __name__ == "__main__": main()
