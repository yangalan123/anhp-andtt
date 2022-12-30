import argparse
import datetime
import glob
import os
import pickle
import numpy as np
import time
import sys
from loguru import logger

import torch
from torch import autograd

from utils.load_synth_data import process_loaded_sequences
from train_functions.train_sahp import make_model, train_eval_sahp, generate_data

DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_SIZE = 16
DEFAULT_LEARN_RATE = 5e-5
DEFAULT_MIN_LENGTH = 50
DEFAULT_MAX_LENGTH = 100
DEFAULT_NUM_SEQS = 800
DEFAULT_EVENT_NUM = 10
DEFAULT_LOOK_AHEAD = 10

parser = argparse.ArgumentParser(description="Generate Data.")
parser.add_argument('-nData', '--num_data', type=int,
                    dest='num_data', default=DEFAULT_NUM_SEQS,
                    help='number of generated dataset size (for training, dev and test are 12.5% of training). (default: {})'.format(DEFAULT_NUM_SEQS))
parser.add_argument('-minlen', type=int,
                    dest='MinLength', default=DEFAULT_MIN_LENGTH,
                    help='minimum length for each generated seq. (default: {})'.format(DEFAULT_MIN_LENGTH))
parser.add_argument('-maxlen', type=int,
                    dest='MaxLength', default=DEFAULT_MAX_LENGTH,
                    help='minimum length for each generated seq. (default: {})'.format(DEFAULT_MAX_LENGTH))
parser.add_argument('-eNum', type=int,
                    dest='EventNum', default=DEFAULT_EVENT_NUM,
                    help='number of event types for each generated seq. (default: {})'.format(DEFAULT_EVENT_NUM))
parser.add_argument('--hidden', type=int,
                    dest='hidden_size', default=DEFAULT_HIDDEN_SIZE,
                    help='number of hidden units. (default: {})'.format(DEFAULT_HIDDEN_SIZE))
parser.add_argument('--look_ahead', type=int,
                    dest='look_ahead', default=DEFAULT_LOOK_AHEAD,
                    help='boundary for computing bounds in adaptive thinning. (default: {})'.format(DEFAULT_LOOK_AHEAD))
parser.add_argument('--d-model', type=int, default=DEFAULT_HIDDEN_SIZE)
parser.add_argument('--atten-heads', type=int, default=8)
parser.add_argument('--pe', type=str,default='add',help='concat, add')
parser.add_argument('--nLayers', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--lambda-l2', type=float, default=3e-4,
                    help='regularization loss.')
parser.add_argument('--dev-ratio', type=float, default=0.1,
                    help='override the size of the dev dataset.')
parser.add_argument('--early-stop-threshold', type=float, default=1e-2,
                    help='early_stop_threshold')
parser.add_argument('--log-dir', type=str,
                    dest='log_dir', default='GenDatalogs',
                    help="training logs target directory.")
parser.add_argument('--save_model', default=True,
                    help="do not save the models state dict and loss history.")
parser.add_argument('--bias', default=False,
                    help="use bias on the activation (intensity) layer.")
parser.add_argument('--samples', default=10,
                    help="number of samples in the integral.")
parser.add_argument('-m', '--model', default='sahp',
                    type=str, choices=['sahp'],
                    help='choose which models to train.')
parser.add_argument('-cp', '--checkpoint', default="model.pt",
                    type=str,
                    help='choose which checkpoint to eval.')
args = parser.parse_args()

if torch.cuda.is_available():
    USE_CUDA = True
else:
    USE_CUDA = False

# SYNTH_DATA_FILES = glob.glob("../data/simulated/*.pkl")
# TYPE_SIZE_DICT = {'retweet': 3, 'bookorder':8, 'meme':5000, 'mimic':75, 'stackOverflow':22,
#                   'synthetic':2}
# REAL_WORLD_TASKS = list(TYPE_SIZE_DICT.keys())[:5]
# SYNTHETIC_TASKS = list(TYPE_SIZE_DICT.keys())[5:]

start_time = time.time()

if __name__ == '__main__':
    # args.log_dir = os.path.join(args.log_dir, )
    os.makedirs(args.log_dir, exist_ok=True)
    # sys.stdout = open(os.path.join(args.log_dir, "log.txt"), "w")
    logger.add(os.path.join(args.log_dir, "log_{time}.log"))
    logger.info(args)
    cuda_num = 'cuda:{}'.format(args.cuda)
    device = torch.device(cuda_num if USE_CUDA else 'cpu')
    logger.info("Training on device {}".format(device))


    train_sample_size = args.num_data
    logger.info("Train sample size: {}".format(train_sample_size))

    # dev_sample_size = dev_seq_times.size(0)
    dev_sample_size = int(args.num_data / 8)
    logger.info("Dev sample size: {}".format(dev_sample_size))

    test_sample_size = int(args.num_data / 8)
    logger.info("Test sample size: {}".format(test_sample_size))



    MODEL_TOKEN = args.model
    logger.info("Chose models {}".format(MODEL_TOKEN))
    hidden_size = args.hidden_size
    logger.info("Hidden size: {}".format(hidden_size))
    # learning_rate = args.lr
    # Training parameters
    # BATCH_SIZE = args.batch_size
    # EPOCHS = args.epochs

    model = None
    d_model = args.d_model
    atten_heads = args.atten_heads
    dropout = args.dropout

    model = make_model(nLayers=args.nLayers, d_model=d_model, atten_heads=atten_heads,
                       dropout=dropout, process_dim=args.EventNum, device=device, pe=args.pe,
                       max_sequence_length=args.MaxLength + 1).to(device)
    total_tokens = 0
    total_patience_counter = 0
    for _split, _num in zip(["train", "test", "dev"], [train_sample_size, test_sample_size, dev_sample_size]):
        save_path = os.path.join(args.log_dir, f"{_split}.pkl")
        with open(save_path, "wb") as f_out:
            data, patience_counter = generate_data(device, model, _num, logger, args)
            pickle.dump(
                {
                    "dim_process": args.EventNum,
                    _split: data
                }, f_out
            )
            total_patience_counter += patience_counter
            total_tokens += sum([len(x) - 1 for x in data])
    print("Impatient Rate: {}".format(total_patience_counter / total_tokens))

    model_save_path = os.path.join(args.log_dir, args.checkpoint)
    torch.save(model.state_dict(), model_save_path)
    # if MODEL_TOKEN == 'sahp':
    #     # with autograd.detect_anomaly(False):
    #         params = args, process_dim, device, tmax, \
    #                  train_times_tensor, train_seq_types, train_seq_lengths, \
    #                  dev_times_tensor, dev_seq_types, dev_seq_lengths, \
    #                  test_times_tensor, test_seq_types, test_seq_lengths, \
    #                  BATCH_SIZE, EPOCHS, USE_CUDA, logger
    #         model = train_eval_sahp(params)
    #
    # else:
    #     exit()


    # if args.save_model:
    #     # Model file dump
    #     SAVED_MODELS_PATH = os.path.abspath(os.path.join(args.log_dir, 'saved_models'))
    #     os.makedirs(SAVED_MODELS_PATH, exist_ok=True)
    #     # print("Saved models directory: {}".format(SAVED_MODELS_PATH))
    #
    #     date_format = "%Y%m%d-%H%M%S"
    #     now_timestamp = datetime.datetime.now().strftime(date_format)
    #     extra_tag = "{}".format(args.task)
    #     filename_base = "{}-{}_hidden{}-{}".format(
    #         MODEL_TOKEN, extra_tag,
    #         hidden_size, now_timestamp)
    #     from utils.save_model import save_model
    #     save_model(model, chosen_file, extra_tag,
    #                hidden_size, now_timestamp, SAVED_MODELS_PATH, MODEL_TOKEN)

    logger.info('Done! time elapsed %.2f sec for %d tokens' % (time.time() - start_time, total_tokens))
    sys.stdout.close()

