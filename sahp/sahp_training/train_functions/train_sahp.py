import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd

import numpy as np
import random

from models.sahp import SAHP
from utils import atten_optimizer
from utils import util
import os

def make_model(nLayers=6, d_model=128, atten_heads=8, dropout=0.1, process_dim=10,
               device = 'cpu', pe='concat', max_sequence_length=4096):
    "helper: construct a models form hyper parameters"

    model = SAHP(nLayers, d_model, atten_heads, dropout=dropout, process_dim=process_dim, device = device,
                 max_sequence_length=max_sequence_length)

    # initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def subsequent_mask(size):
    "mask out subsequent positions"
    atten_shape = (1,size,size)
    # np.triu: Return a copy of a matrix with the elements below the k-th diagonal zeroed.
    mask = np.triu(np.ones(atten_shape),k=1).astype('uint8')
    aaa = torch.from_numpy(mask) == 0
    return aaa


class MaskBatch():
    "object for holding a batch of data with mask during training"
    def __init__(self,src,pad, device):
        self.src = src
        self.src_mask = self.make_std_mask(self.src, pad, device)

    @staticmethod
    def make_std_mask(tgt,pad,device):
        "create a mask to hide padding and future input"
        # torch.cuda.set_device(device)
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)).to(device)
        return tgt_mask


def l1_loss(model):
    ## l1 loss
    l1 = 0
    for p in model.parameters():
        l1 = l1 + p.abs().sum()
    return l1

def eval_sahp(batch_size, loop_range, seq_lengths, seq_times, seq_types, model, device, lambda_l1=0):
    model.eval()
    epoch_loss = 0
    for i_batch in loop_range:
        batch_onehot, batch_seq_times, batch_dt, batch_seq_types, _, _, _, batch_seq_lengths = \
            util.get_batch(batch_size, i_batch, model, seq_lengths, seq_times, seq_types, rnn=False)
        batch_seq_types_ = batch_seq_types[:, 1:]

        masked_seq_types = MaskBatch(batch_seq_types_, pad=model.process_dim, device=device)# exclude the first added event
        model.forward(batch_seq_times[:, :-1], batch_seq_types[:, :-1], masked_seq_types.src_mask)
        nll, _, _, _ = model.compute_loss(batch_seq_times, batch_onehot)

        loss = nll
        epoch_loss += loss.detach()
    event_num = torch.sum(seq_lengths).float()
    model.train()
    return event_num, epoch_loss


def get_stats(loop_range, batch_size, model, seq_lengths, seq_times, seq_types, device):
    ret_log_intens = []
    ret_log_intens_ret = []
    ret_integrals = []
    for i_batch in loop_range:
        # model_opt.optimizer.zero_grad()

        batch_onehot, batch_seq_times, batch_dt, batch_seq_types, _, _, _, batch_seq_lengths = \
            util.get_batch(batch_size, i_batch, model, seq_lengths, seq_times, seq_types, rnn=False)

        batch_seq_types_ = batch_seq_types[:, 1:]

        masked_seq_types = MaskBatch(batch_seq_types_, pad=model.process_dim,
                                     device=device)  # exclude the first added even
        model.forward(batch_seq_times[:, :-1], batch_seq_types[:, :-1], masked_seq_types.src_mask)
        nll, log_intens, log_intens_ret, integrals = model.compute_loss(batch_seq_times, batch_onehot)
        ret_log_intens.extend([x for x in log_intens.detach().cpu()])
        ret_log_intens_ret.extend([x for x in log_intens_ret.detach().cpu()])
        ret_integrals.extend([x for x in integrals.detach().cpu()])
        assert log_intens_ret.size(1) == integrals.size(1)
    return ret_log_intens, ret_log_intens_ret, ret_integrals



def train_eval_sahp(params):

    args, process_dim, device, tmax, \
    train_seq_times, train_seq_types, train_seq_lengths, \
    dev_seq_times, dev_seq_types, dev_seq_lengths, \
    test_seq_times, test_seq_types, test_seq_lengths, \
    batch_size, epoch_num, use_cuda, logger = params

    ## sequence length
    train_seq_lengths, reorder_indices_train = train_seq_lengths.sort(descending=True)
    # # Reorder by descending sequence length
    train_seq_times = train_seq_times[reorder_indices_train]
    train_seq_types = train_seq_types[reorder_indices_train]
    #
    dev_seq_lengths, reorder_indices_dev = dev_seq_lengths.sort(descending=True)
    # # Reorder by descending sequence length
    dev_seq_times = dev_seq_times[reorder_indices_dev]
    dev_seq_types = dev_seq_types[reorder_indices_dev]

    test_seq_lengths, reorder_indices_test = test_seq_lengths.sort(descending=True)
    # # Reorder by descending sequence length
    test_seq_times = test_seq_times[reorder_indices_test]
    test_seq_types = test_seq_types[reorder_indices_test]

    max_sequence_length = max(train_seq_lengths[0], dev_seq_lengths[0], test_seq_lengths[0])
    logger.info('max_sequence_length: {}'.format(max_sequence_length))

    d_model = args.d_model
    atten_heads = args.atten_heads
    dropout = args.dropout

    model = make_model(nLayers=args.nLayers, d_model=d_model, atten_heads=atten_heads,
                    dropout=dropout, process_dim=process_dim, device=device, pe=args.pe,
                    max_sequence_length=max_sequence_length + 1).to(device)

    logger.info("the number of trainable parameters: " + str(util.count_parameters(model)))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.lambda_l2)
    model_opt = atten_optimizer.NoamOpt(args.d_model, 1, 100, initial_lr=args.lr, optimizer=optimizer)


    ## Size of the traing dataset
    train_size = train_seq_times.size(0)
    dev_size = dev_seq_times.size(0)
    test_size = test_seq_times.size(0)
    tr_loop_range = list(range(0, train_size, batch_size))
    de_loop_range = list(range(0, dev_size, batch_size))
    test_loop_range = list(range(0, test_size, batch_size))

    last_dev_loss = 0.0
    early_step = 0

    model.train()
    for epoch in range(epoch_num):
        epoch_train_loss = 0.0
        logger.info('Epoch {} starts '.format(epoch))

        ## training
        random.shuffle(tr_loop_range)
        for i_batch in tr_loop_range:

            model_opt.optimizer.zero_grad()

            batch_onehot, batch_seq_times, batch_dt, batch_seq_types, _, _, _, batch_seq_lengths = \
                util.get_batch(batch_size, i_batch, model, train_seq_lengths, train_seq_times, train_seq_types, rnn=False)

            batch_seq_types_ = batch_seq_types[:, 1:]

            masked_seq_types = MaskBatch(batch_seq_types_, pad=model.process_dim, device=device)# exclude the first added even
            model.forward(batch_seq_times[:, :-1], batch_seq_types[:, :-1], masked_seq_types.src_mask)
            nll, _, _, _ = model.compute_loss(batch_seq_times, batch_onehot)

            loss = nll

            loss.backward()
            model_opt.optimizer.step()

            if i_batch %50 == 0:
                batch_event_num = torch.sum(batch_seq_lengths).float()
                logger.info('Epoch {} Batch {}: Negative Log-Likelihood per event: {:5f} nats' \
                      .format(epoch, i_batch, loss.item()/ batch_event_num))
            epoch_train_loss += loss.detach()

        if epoch_train_loss < 0:
            break
        train_event_num = torch.sum(train_seq_lengths).float()
        # - train_size : starting from first event
        logger.info('---\nEpoch.{} Training set\nTrain Negative Log-Likelihood per event: {:5f} nats\n' \
              .format(epoch, -epoch_train_loss / (train_event_num) ))

        ## dev
        dev_event_num, epoch_dev_loss = eval_sahp(batch_size, de_loop_range, dev_seq_lengths, dev_seq_times,
                                                 dev_seq_types, model, device, args.lambda_l2)
        logger.info('Epoch.{} Devlopment set\nDev Negative Likelihood per event: {:5f} nats.\n'. \
              format(epoch, -epoch_dev_loss / (dev_event_num) ))

        ## test
        test_event_num, epoch_test_loss = eval_sahp(batch_size, test_loop_range, test_seq_lengths, test_seq_times,
                                                   test_seq_types, model, device, args.lambda_l2)
        logger.info('Epoch.{} Test set\nTest Negative Likelihood per event: {:5f} nats.\n'. \
              format(epoch, -epoch_test_loss / (test_event_num) ))

        ## early stopping
        gap = epoch_dev_loss / dev_event_num - last_dev_loss
        if abs(gap) < args.early_stop_threshold:
            early_step += 1
        last_dev_loss = epoch_dev_loss / dev_event_num

        if early_step >=3:
            logger.info('Early Stopping')
            break

        # prediction
        # avg_rmse, types_predict_score = \
        #     prediction_evaluation(device, model, test_seq_lengths, test_seq_times, test_seq_types, test_size, tmax, logger)

    test_log_intens, test_log_intens_ret, test_integral = get_stats(test_loop_range, batch_size, model, test_seq_lengths, test_seq_times, test_seq_types, device)
    torch.save({
        "log_intens": test_log_intens,
        "log_intens_ret": test_log_intens_ret,
        "integral": test_integral,
        "seq_lengths": test_seq_lengths,
        "types": test_seq_types,
        "times": test_seq_times
    }, os.path.join(args.log_dir, "test_stat.pt"))

    return model


def prediction_evaluation(device, model, test_seq_lengths, test_seq_times, test_seq_types, test_size, tmax, logger):
    model.eval()
    from utils import evaluation
    test_data = (test_seq_times, test_seq_types, test_seq_lengths)
    incr_estimates, incr_errors, types_real, types_estimates = \
        evaluation.predict_test(model, *test_data, pad=model.process_dim, device=device,
                                hmax=tmax, use_jupyter=False, rnn=False)
    if device != 'cpu':
        incr_errors = [incr_err.item() for incr_err in incr_errors]
        types_real = [types_rl.item() for types_rl in types_real]
        types_estimates = [types_esti.item() for types_esti in types_estimates]

    avg_rmse = np.sqrt(np.mean(incr_errors), dtype=np.float64)
    logger.info("rmse: {}".format(avg_rmse))
    mse_var = np.var(incr_errors, dtype=np.float64)

    delta_meth_stderr = 1 / test_size * mse_var / (4 * avg_rmse)

    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
    # types_predict_score = f1_score(types_real, types_estimates, average='micro')# preferable in class imbalance
    types_predict_score = accuracy_score(types_real, types_estimates)# preferable in class imbalance
    logger.info("Type prediction score: {}".format(types_predict_score))
    # print("Confusion matrix:\n", confusion_matrix(types_real, types_estimates))
    model.train()
    return avg_rmse, types_predict_score

if __name__ == "__main__":
    mode = 'train'

    if mode == 'train':
        with autograd.detect_anomaly():
            train_eval_sahp()

    else:
        pass
    print("Done!")



