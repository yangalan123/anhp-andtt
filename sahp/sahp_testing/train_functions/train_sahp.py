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
from snhp.eval.sigtest import Bootstrapping
import os
import random
from tqdm import tqdm


def make_model(nLayers=6, d_model=128, atten_heads=8, dropout=0.1, process_dim=10,
               device='cpu', pe='concat', max_sequence_length=4096):
    "helper: construct a models form hyper parameters"

    model = SAHP(nLayers, d_model, atten_heads, dropout=dropout, process_dim=process_dim, device=device,
                 max_sequence_length=max_sequence_length)

    # initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def subsequent_mask(size):
    "mask out subsequent positions"
    atten_shape = (1, size, size)
    # np.triu: Return a copy of a matrix with the elements below the k-th diagonal zeroed.
    mask = np.triu(np.ones(atten_shape), k=1).astype('uint8')
    aaa = torch.from_numpy(mask) == 0
    return aaa


class MaskBatch():
    "object for holding a batch of data with mask during training"

    def __init__(self, src, pad, device):
        self.src = src
        self.src_mask = self.make_std_mask(self.src, pad, device)

    @staticmethod
    def make_std_mask(tgt, pad, device):
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

        masked_seq_types = MaskBatch(batch_seq_types_, pad=model.process_dim,
                                     device=device)  # exclude the first added event
        model.forward(batch_seq_times[:, :-1], batch_seq_types[:, :-1], masked_seq_types.src_mask)
        nll, _, _, _ = model.compute_loss(batch_seq_times, batch_onehot)

        loss = nll
        epoch_loss += loss.detach()
    event_num = torch.sum(seq_lengths).float()
    model.train()
    return event_num, epoch_loss


def get_stats(loop_range, batch_size, model, seq_lengths, seq_times, seq_types, device):
    ret_intens = []
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
    return ret_intens, ret_log_intens_ret, ret_integrals


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
    model.load_state_dict(torch.load(os.path.join(args.log_dir, args.checkpoint)))

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
                util.get_batch(batch_size, i_batch, model, train_seq_lengths, train_seq_times, train_seq_types,
                               rnn=False)

            batch_seq_types_ = batch_seq_types[:, 1:]

            masked_seq_types = MaskBatch(batch_seq_types_, pad=model.process_dim,
                                         device=device)  # exclude the first added even
            model.forward(batch_seq_times[:, :-1], batch_seq_types[:, :-1], masked_seq_types.src_mask)
            nll, _, _, _ = model.compute_loss(batch_seq_times, batch_onehot)

            loss = nll

            loss.backward()
            model_opt.optimizer.step()

            if i_batch % 50 == 0:
                batch_event_num = torch.sum(batch_seq_lengths).float()
                logger.info('Epoch {} Batch {}: Negative Log-Likelihood per event: {:5f} nats' \
                            .format(epoch, i_batch, loss.item() / batch_event_num))
            epoch_train_loss += loss.detach()

        if epoch_train_loss < 0:
            break
        train_event_num = torch.sum(train_seq_lengths).float()
        # - train_size : starting from first event
        logger.info('---\nEpoch.{} Training set\nTrain Negative Log-Likelihood per event: {:5f} nats\n' \
                    .format(epoch, -epoch_train_loss / (train_event_num)))

        ## dev
        dev_event_num, epoch_dev_loss = eval_sahp(batch_size, de_loop_range, dev_seq_lengths, dev_seq_times,
                                                  dev_seq_types, model, device, args.lambda_l2)
        logger.info('Epoch.{} Devlopment set\nDev Negative Likelihood per event: {:5f} nats.\n'. \
                    format(epoch, -epoch_dev_loss / (dev_event_num)))

        ## test
        test_event_num, epoch_test_loss = eval_sahp(batch_size, test_loop_range, test_seq_lengths, test_seq_times,
                                                    test_seq_types, model, device, args.lambda_l2)
        logger.info('Epoch.{} Test set\nTest Negative Likelihood per event: {:5f} nats.\n'. \
                    format(epoch, -epoch_test_loss / (test_event_num)))

        ## early stopping
        gap = epoch_dev_loss / dev_event_num - last_dev_loss
        if abs(gap) < args.early_stop_threshold:
            early_step += 1
        last_dev_loss = epoch_dev_loss / dev_event_num

        if early_step >= 3:
            logger.info('Early Stopping')
            break

        # prediction
        # avg_rmse, types_predict_score = \
        #     prediction_evaluation(device, model, test_seq_lengths, test_seq_times, test_seq_types, test_size, tmax, logger)

    token_errs, token_mses, token_logliks, token_type_lls, token_time_lls = prediction_evaluation(device, model,
                                                                                                  test_seq_lengths,
                                                                                                  test_seq_times,
                                                                                                  test_seq_types,
                                                                                                  test_size, tmax,
                                                                                                  logger)
    bs = Bootstrapping()
    for arr, name in zip([token_errs, token_mses, token_logliks, token_type_lls, token_time_lls],
                         ["error_rate", "mse", "loglik", "type_ll", "time_ll"]):
        # metric = sum([x[0] for x in arr]) / sum([x[1] for x in arr])
        metric_mean = np.mean([x[0] for x in arr])
        metric_std = np.std([x[0] for x in arr])
        # if name == "mse":
        #     metric_mean = np.sqrt(metric_mean)
        #     metric_std = np.sqrt(metric_std)
        print(f"{name}: {metric_mean} ({metric_std})")
        bs.run(arr)
    # test_intens, test_log_intens_ret, test_integral = get_stats(test_loop_range, batch_size, model, test_seq_lengths, test_seq_times, test_seq_types, device)
    # torch.save({
    #     "log_intens": test_intens,
    #     "log_intens_ret": test_log_intens_ret,
    #     "integral": test_integral,
    #     "seq_lengths": test_seq_lengths,
    #     "types": test_seq_types,
    #     "times": test_seq_times
    # }, os.path.join(args.log_dir, "test_stat.pt"))

    return model


# def prediction_evaluation(device, model, test_seq_lengths, test_seq_times, test_seq_types, test_size, tmax, logger):
#     model.eval()
#     from utils import evaluation
#     test_data = (test_seq_times, test_seq_types, test_seq_lengths)
#     incr_estimates, incr_errors, types_real, types_estimates = \
#         evaluation.predict_test(model, *test_data, pad=model.process_dim, device=device,
#                                 hmax=tmax, use_jupyter=False, rnn=False)
#     if device != 'cpu':
#         incr_errors = [incr_err.item() for incr_err in incr_errors]
#         types_real = [types_rl.item() for types_rl in types_real]
#         types_estimates = [types_esti.item() for types_esti in types_estimates]
#
#     avg_rmse = np.sqrt(np.mean(incr_errors), dtype=np.float64)
#     logger.info("rmse: {}".format(avg_rmse))
#     mse_var = np.var(incr_errors, dtype=np.float64)
#
#     delta_meth_stderr = 1 / test_size * mse_var / (4 * avg_rmse)
#
#     from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
#     # types_predict_score = f1_score(types_real, types_estimates, average='micro')# preferable in class imbalance
#     types_predict_score = accuracy_score(types_real, types_estimates)# preferable in class imbalance
#     logger.info("Type prediction score: {}".format(types_predict_score))
#     # print("Confusion matrix:\n", confusion_matrix(types_real, types_estimates))
#     model.train()
#     return avg_rmse, types_predict_score
def generate_data(device, model, num_data, logger, args):
    model.eval()
    from utils.load_synth_data import one_hot_embedding
    res = []
    patience_counter = 0
    with torch.no_grad():
        for i in tqdm(range(num_data)):
            initial_event_type = random.sample(range(model.process_dim), 1)[0]
            batch_seq_types_list = [initial_event_type, ]
            batch_seq_types = torch.tensor(batch_seq_types_list).reshape(1, 1).to(device)
            batch_time_list = [0, ]
            batch_dtime = [0, ]
            length = random.randint(args.MinLength, args.MaxLength)
            batch_seq_times = torch.Tensor(batch_time_list).reshape(1, 1).to(device)
            batch_intens = [[1 / model.process_dim for _ in range(model.process_dim)], ]
            for j in range(length):
                masked_seq_types = MaskBatch(batch_seq_types, pad=model.process_dim,
                                             device=device)  # exclude the first added even
                model.forward(batch_seq_times, batch_seq_types, masked_seq_types.src_mask)
                current_time = batch_time_list[-1]
                time_pred, _patience_counter = thinning_unbiased(device, model, current_time, args.look_ahead, j)
                patience_counter += _patience_counter
                batch_time_list.append(time_pred)
                batch_dtime.append(time_pred - current_time)
                batch_seq_times = torch.Tensor(batch_time_list).reshape(1, -1).to(device)

                intens = model.compute_intens_at_given_time(batch_seq_times)
                next_event_name = torch.multinomial(intens[0, -1], 1)[0].item()
                assert len(intens.size()) == 3
                batch_seq_types_list.append(next_event_name)
                batch_seq_types = torch.tensor(batch_seq_types_list).reshape(1, -1).to(device)
                batch_intens.append(intens[0, -1].tolist())
            ret = [
                {
                    "time_since_start": batch_time_list[x],
                    "type_event": batch_seq_types_list[x],
                    "time_since_last_event": batch_dtime[x],
                    "intensities": batch_intens[x]
                }
                for x in range(len(batch_time_list))
            ]
            res.append(ret)
    return res, patience_counter


def prediction_evaluation(device, model, test_seq_lengths, test_seq_times, test_seq_types, test_size, tmax, logger):
    model.eval()
    token_errs = []
    token_mses = []
    token_type_lls = []
    token_time_lls = []
    token_logliks = []
    with torch.no_grad():
        for i in range(test_size):
            # seq_types, seq_times, lengths = test_seq_types[i: i + 1], test_seq_times[i: i + 1]
            batch_onehot, batch_seq_times, batch_dt, batch_seq_types, _, _, _, batch_seq_lengths = \
                util.get_batch(1, i, model, test_seq_lengths, test_seq_times, test_seq_types, rnn=False)

            batch_seq_types_ = batch_seq_types[:, 1:]

            masked_seq_types = MaskBatch(batch_seq_types_, pad=model.process_dim,
                                         device=device)  # exclude the first added even
            model.forward(batch_seq_times[:, :-1], batch_seq_types[:, :-1], masked_seq_types.src_mask)
            nll, intens, log_intens_ret, integrals = model.compute_loss(batch_seq_times, batch_onehot)
            assert intens.size(0) == 1
            # log_intens = log_intens[0]
            for j in range(1, batch_seq_lengths.item() - 1):
                token_errs.append((torch.argmax(intens[0][j], dim=0).item() != batch_seq_types[:, j + 1].item(), 1.0))
                token_loglik = (log_intens_ret[0][j] - integrals[0][j]).item()
                token_type_ll = (log_intens_ret[0][j] - intens[0][j].sum(dim=-1).log()).item()
                token_type_lls.append(
                    (token_type_ll, 1.0)
                )
                token_time_lls.append(
                    (token_loglik - token_type_ll, 1.0)
                )
                token_logliks.append((token_loglik, 1.0))
                current_time = batch_seq_times[0][j]
                next_time = batch_seq_times[0][j + 1]
                time_pred = thinning(device, model, current_time, next_time, j)
                token_mses.append(((time_pred - next_time.item()) ** 2, 1.0))
                # token_mses.append((abs(time_pred - current_time.item()), 1.0))
    return token_errs, token_mses, token_logliks, token_type_lls, token_time_lls


def thinning(device, model, current_time, next_time, index, over_sample_rate=5.0, num_samples_for_bound=10, num_exp=100,
             num_samples=1):
    # implement adaptive thinning
    # compute the estimated look_ahead
    look_ahead = 4 * (next_time - current_time)
    boundary = current_time + look_ahead
    rst = torch.empty(
        size=[num_samples], dtype=torch.float32, device=device
    ).fill_(boundary)
    # for those didn't accept proposed times, use boundary or even farther
    weights = torch.ones(
        size=[num_samples], dtype=torch.float32, device=device)
    weights /= weights.sum()
    # 1. compute the adaptive bound
    # 1.1 sample dtimes for bound computation
    sampled_dtimes_for_bound = torch.rand(num_samples_for_bound, 1).to(device) * look_ahead
    # [1, dim]
    start_point = model.start_point[:, index]
    converge_point = model.converge_point[:, index]
    omega = model.omega[:, index]
    sampled_hiddens_for_bound = model.state_decay(converge_point, start_point, omega, sampled_dtimes_for_bound)
    sampled_intensities = model.intensity_layer(sampled_hiddens_for_bound)
    # sampled_intensities = torch.ones_like(sampled_intensities) * 1 / 50
    # [num_samples_for_bound, dim] -> [num_samples_for_bound, event_num]
    bound = sampled_intensities.sum(dim=-1).max() * over_sample_rate
    # 2. sampled times and compute intensities
    sample_rate = bound
    Exp_numbers = torch.empty(size=[num_samples, num_exp], dtype=torch.float32, device=device)
    Exp_numbers.exponential_(1.0)
    sampled_times = Exp_numbers / sample_rate
    sampled_times = sampled_times.cumsum(dim=-1) + current_time
    # min_time_per_row, _ = sampled_times.min(dim=-1)
    # accepted_rows = min_time_per_row < boundary
    # sampled_times = sampled_times[accepted_rows]

    sampled_dtimes = sampled_times - current_time
    hiddens_at_sampled_times = model.state_decay(converge_point.unsqueeze(0), start_point.unsqueeze(0),
                                                 omega.unsqueeze(0), sampled_dtimes.unsqueeze(-1))
    intensities_at_sampled_times = model.intensity_layer(hiddens_at_sampled_times)
    # intensities_at_sampled_times = torch.ones_like(intensities_at_sampled_times) * 1 / 50
    total_intensities = intensities_at_sampled_times.sum(dim=-1)
    cur_sample_num = sampled_times.size(0)
    Unif_numbers = torch.empty(size=[cur_sample_num, num_exp], dtype=torch.float32, device=device)
    Unif_numbers.uniform_(0.0, 1.0)
    criterion = Unif_numbers * sample_rate / total_intensities

    """
    for each parallel draw, find its min criterion
    if that < 1.0, the 1st (i.e. smallest) sampled time with cri < 1.0 is accepted 
    if none is accepted, use boundary/maxsampletime for that draw
    """
    min_cri_each_draw, _ = criterion.min(dim=1)
    who_has_accepted_times = min_cri_each_draw < 1.0
    # print(f"who has accepted times : {who_has_accepted_times}")
    """
    whoever accepts times, find their accepted times
    """
    sampled_times_accepted = sampled_times.clone()
    sampled_times_accepted[criterion >= 1.0] = sampled_times.max() + 1.0
    accepted_times_each_draw, accepted_id_each_draw = sampled_times_accepted.min(dim=-1)
    # size : N
    rst[who_has_accepted_times] = \
        accepted_times_each_draw[who_has_accepted_times]
    who_not_accept = ~who_has_accepted_times
    who_reach_further = sampled_times[:, -1] > boundary
    rst[who_not_accept & who_reach_further] = \
        sampled_times[:, -1][who_not_accept & who_reach_further]
    return torch.sum(rst * weights).item()


def thinning_unbiased(device, model, current_time, boundary, index, over_sample_rate=5.0,
                      num_samples_for_bound=10,
                      num_exp=100, num_samples=1, patience=5):
    # implement adaptive thinning
    # compute the estimated look_ahead
    # look_ahead = 4 * (next_time - current_time)
    # boundary = current_time + look_ahead
    rst = torch.empty(
        size=[num_samples], dtype=torch.float32, device=device
    ).fill_(boundary)
    # for those didn't accept proposed times, use boundary or even farther
    weights = torch.ones(
        size=[num_samples], dtype=torch.float32, device=device)
    weights /= weights.sum()
    look_ahead = boundary - current_time
    target_num_samples = num_samples
    time_last_event = current_time
    patience_counter = 0
    while target_num_samples > 0 and patience > 0:
        # 1. compute the adaptive bound
        # 1.1 sample dtimes for bound computation
        sampled_dtimes_for_bound = torch.rand(num_samples_for_bound, 1).to(device) * (boundary - time_last_event)
        # add up the shifted value as we move to next sample interval
        sampled_dtimes_for_bound += time_last_event - current_time
        # [1, dim]
        start_point = model.start_point[:, index]
        converge_point = model.converge_point[:, index]
        omega = model.omega[:, index]
        sampled_hiddens_for_bound = model.state_decay(converge_point, start_point, omega, sampled_dtimes_for_bound)
        sampled_intensities = model.intensity_layer(sampled_hiddens_for_bound)
        # sampled_intensities = torch.ones_like(sampled_intensities) * 1 / 50
        # [num_samples_for_bound, dim] -> [num_samples_for_bound, event_num]
        bound = sampled_intensities.sum(dim=-1).max() * over_sample_rate
        # 2. sampled times and compute intensities
        sample_rate = bound
        Exp_numbers = torch.empty(size=[num_samples, num_exp], dtype=torch.float32, device=device)
        Exp_numbers.exponential_(1.0)
        sampled_times = Exp_numbers / sample_rate
        # sampled_times = sampled_times.cumsum(dim=-1) + current_time
        sampled_times = sampled_times.cumsum(dim=-1) + time_last_event
        # min_time_per_row, _ = sampled_times.min(dim=-1)
        # accepted_rows = min_time_per_row < boundary
        # sampled_times = sampled_times[accepted_rows]

        sampled_dtimes = sampled_times - current_time
        hiddens_at_sampled_times = model.state_decay(converge_point.unsqueeze(0), start_point.unsqueeze(0),
                                                     omega.unsqueeze(0), sampled_dtimes.unsqueeze(-1))
        intensities_at_sampled_times = model.intensity_layer(hiddens_at_sampled_times)
        # intensities_at_sampled_times = torch.ones_like(intensities_at_sampled_times) * 1 / 50
        total_intensities = intensities_at_sampled_times.sum(dim=-1)
        cur_sample_num = sampled_times.size(0)
        Unif_numbers = torch.empty(size=[cur_sample_num, num_exp], dtype=torch.float32, device=device)
        Unif_numbers.uniform_(0.0, 1.0)
        criterion = Unif_numbers * sample_rate / total_intensities

        """
        for each parallel draw, find its min criterion
        if that < 1.0, the 1st (i.e. smallest) sampled time with cri < 1.0 is accepted 
        if none is accepted, use boundary/maxsampletime for that draw
        """
        min_cri_each_draw, _ = criterion.min(dim=1)
        who_has_accepted_times = min_cri_each_draw < 1.0
        """
        whoever accepts times, find their accepted times
        """
        sampled_times_accepted = sampled_times.clone()
        sampled_times_accepted[criterion >= 1.0] = sampled_times.max() + 1.0
        accepted_times_each_draw, accepted_id_each_draw = sampled_times_accepted.min(dim=-1)
        accepted_sample_num_this_round = min(target_num_samples, who_has_accepted_times.sum())
        if accepted_sample_num_this_round > 0:
            if accepted_sample_num_this_round < target_num_samples:
                rst[-target_num_samples: -target_num_samples + accepted_sample_num_this_round] = \
                    accepted_times_each_draw[who_has_accepted_times][:accepted_sample_num_this_round]
            else:
                rst[-target_num_samples:] = \
                    accepted_times_each_draw[who_has_accepted_times][:accepted_sample_num_this_round]

        time_last_event += min(sampled_times.max().item() - time_last_event, look_ahead)
        boundary = time_last_event + look_ahead
        target_num_samples -= accepted_sample_num_this_round
        # num_samples_for_bounds += delta_num_samples_for_bounds
        if accepted_sample_num_this_round == 0:
            patience -= 1
        if target_num_samples > 0:
            rst[-target_num_samples:] = boundary
    if patience == 0:
        patience_counter += 1
    return torch.sum(rst * weights).item(), patience_counter

# if __name__ == "__main__":
#     mode = 'train'
#
#     if mode == 'train':
#         with autograd.detect_anomaly():
#             train_eval_sahp()
#
#     else:
#         pass
#     print("Done!")
