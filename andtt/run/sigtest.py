# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
significant test
@author: hongyuan
"""

import pickle
import time
import numpy
import random
import os
import datetime
from itertools import chain

from ndtt.eval.sigtest import Bootstrapping, PairPerm
from ndtt.eval.draw import Drawer

from ndtt.functions.utils import getAbsPath

import argparse
__author__ = 'Hongyuan Mei'


def main():

    parser = argparse.ArgumentParser(description='significant test')


    parser.add_argument(
        '-d', '--Domain', required=True, type=str, help='which domain to work on?'
    )
    parser.add_argument(
        '-fn', '--FolderName', required=True, type=str,
        help='base name of the folder to store result?'
    )
    parser.add_argument(
        '-s', '--Split', required=True, type=str, help='what split to test?',
        choices = ['train', 'dev', 'test']
    )
    parser.add_argument(
        '-c', '--Compare', action='store_true', help='compare with another model?'
    )
    parser.add_argument(
        '-fn2', '--FolderName2', type=str,
        help='base name of the folder to store another result?'
    )
    parser.add_argument(
        '-df', '--DrawFigure', action='store_true', help='draw figure?'
    )
    parser.add_argument(
        '-ps', '--PathStorage', type=str, default='../..',
        help='Path of storage which stores domains (with data), logs, results, etc. \
            Must be local (e.g. no HDFS allowed)'
    )
    parser.add_argument(
        '-sd', '--Seed', default=12345, type=int, help='random seed'
    )

    args = parser.parse_args()

    path_storage = getAbsPath(args.PathStorage)
    args.PathDomain = os.path.join(path_storage, 'domains', args.Domain)

    bs = Bootstrapping()

    if args.Domain == args.FolderName:
        args.Database = 'gen'
        args.PathResult = os.path.join(args.PathDomain, f'results_gen_{args.Split}')
    else:
        path_logs = os.path.join( args.PathDomain, 'Logs', args.FolderName )
        assert os.path.exists(path_logs)
        args.PathResult = os.path.join(path_logs, f'results_{args.Split}')
    with open(args.PathResult, 'rb') as f:
        allres = pickle.load(f)

    bs.run(allres['loglik'])

    if args.Compare:
        """
        need to compare with another result by paired perm test
        """
        if args.Domain == args.FolderName2:
            args.Database2 = 'gen'
            args.PathResult2 = os.path.join(args.PathDomain, f'results_gen_{args.Split}')
        else:
            path_logs = os.path.join( args.PathDomain, 'Logs', args.FolderName2 )
            assert os.path.exists(path_logs)
            args.PathResult2 = os.path.join(path_logs, f'results_{args.Split}')
        with open(args.PathResult2, 'rb') as f:
            allres2 = pickle.load(f)

        print()
        bs.run(allres2['loglik'])

        print()
        pp = PairPerm()
        pp.run(allres['loglik'], allres2['loglik'])

        if args.DrawFigure:
            """
            draw figure to compare two models on same data
            """
            dr = Drawer()

            def _getName(folder_name):
                if 'single' in folder_name:
                    return 'NHP'
                elif 'struct' in folder_name:
                    return 'structured NHP'
                else:
                    raise Exception(f"Unparsable name {folder_name}")

            path_save = os.path.join(args.PathDomain, 'figures')
            if not os.path.exists(path_save):
                os.makedirs(path_save)

            dr.draw(allres['loglik'], allres2['loglik'],
                name1=_getName(args.PathResult),
                name2=_getName(args.PathResult2),
                figname=f"{args.FolderName}_vs_{args.FolderName2}",
                path_save=path_save)


if __name__ == "__main__": main()
