# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
helpers for training and testing

@author: Chenghao Yang (based on Hongyuan's code)
"""

import random
import time

import numpy
import torch

from andtt.neural.cell import TransformerCell

# from tqdm import tqdm
# from anhp.neural.gate import CTLSTMGate

"""
NOTE : make sure that the output of these functions do NOT have computation graph attached
"""
"""
NOTE 
many methods here are almost identical to those in snhp.esm.helpers
we don't achieve 100% code reuse for the sake of future-edit flexibility
i.e., we may modify those functions in the future, 
so decoupling them at the beginning might be a good design, 
at the cost of less code reuse
"""
"""
NOTE 
AHA! here might be a better way!
we define new method, which may or may not call old method 
(1) code is fully reused
(2) it is alway easy to decouple them and modify the new functions in the future
E.g., 
from anhp.esm.helpers import get_subseq as get_subseq_old
def get_subseq(data, split, seq_id): 
    return get_subseq_old(data, split, seq_id)
OR
import snhp.esm.helpers as helpers_old
def get_subseq(data, split, seq):
    return helpers_old.get_subseq(data, split, seq_id)
"""


# @profile
def get_subseq(data, split, seq_id):
    # no MP, subseq == seq
    assert data.seq[split][seq_id][-1]['name'] == 'eos', "not terminated by eos?"
    return data.seq[split][seq_id]


# @profile
def get_logprob_and_grads(arguments):
    log_prob, num_event, _, num_inten = get_logprob_with_graph(arguments)
    loss = -log_prob
    loss.backward()
    rst = float(log_prob)
    del log_prob
    del loss
    return rst, num_event, num_inten


# @profile
def get_logprob_and_grads_cp(arguments, checkpoint_pack):
    # for checkpoint training 
    log_prob, num_event, _, num_inten = get_logprob_with_graph(arguments, checkpoint_pack)
    return log_prob, num_event, num_inten


# @profile
def get_logprob(arguments):
    with torch.no_grad():
        # don't compute gradients 
        # save memory for eval 
        log_prob, num_event, all_log_prob, _ = get_logprob_with_graph(arguments)
        rst = float(log_prob)
        del log_prob
    return rst, num_event, all_log_prob


# @profile
def get_logprob_with_graph(arguments, checkpoint_pack=None):
    """
    unpacked objects should match get_arguments_for_logprob in manager
    """
    seq_id, split, \
    seq, cutoff_time, \
    datalog, down_sampler, \
    multiplier, eps, device, memory_size_per_cell = arguments

    if checkpoint_pack is not None:
        checkpoint, optim, batchsize = checkpoint_pack
        assert checkpoint > 0, \
            f"check point training even for {checkpoint}?!"
        assert optim is not None, "no optimizer?!"
    else:
        # even during testing, we still wish to occasionally release the memory consumption
        checkpoint = max(datalog.args.MemorySize, 1)

    duration = seq[-1]['time']

    assert cutoff_time >= 0.0, "cutoff time must be positive"
    assert cutoff_time < duration, "we should cut off before eos"

    time_last_event = 0.0
    """
    sum_log_intensity and integral : 
    used by entire seq training and output
    sum_log_intensity_cp, integral_cp : 
    used by checkpoint training 
    either of them is tracked in training, not both, to save memory
    """
    sum_log_intensity = 0.0
    integral = 0.0
    sum_log_intensity_cp = 0.0
    integral_cp = 0.0
    """
    all_log_prob : 
    used to track the log-probability of each event
    useful for token-level bootstrapping for eval
    """
    all_log_prob = list()

    num_sub = len(seq) - 1  # not count BOS, i.e. # intervals
    num_event = 0.0  # # of tokens, not including BOS or EOS
    num_inten = 0.0  # # of intensities computed

    update_cdb = False

    """
    cell blocks for this sequence 
    """
    active = {  # use dict for indexing
        'cells': dict(), 'intensity_cells': dict(),
        # 'gates': dict(), 'intensity_gates': dict()
    }
    dead = {  # use list since we don't use dead cells
        'cells': list(), 'intensity_cells': list(),
        # 'gates': list(), 'intensity_gates': list()
    }

    tdb = datalog.tdb[split][seq_id]  # temporal database
    cdb = tdb[-1]['db_after_create']  # init database

    """
    NOTE
    could we safely delete torch.autograd.set_detect_anomaly() now?
    """
    # log_debug_test = open("/home-1/cyang91@jhu.edu/scratch/Transformer-SNHP/marcc_job_manager/debug_test_performance_iptv_log", "w", buffering=1)
    # seq_bar = tqdm(seq, desc="running_with_graph")
    with torch.autograd.set_detect_anomaly(False):
        for i, event in enumerate(seq):  # loop over seq
            ### debug only ###
            # if i == 5:
            #     break
            """
            use current database to compute log probability of this event 
            because we want prob of this event happening given current facts
            """
            # print(f"{i}-th event : {event}")
            """
            prepare times and types for which to compute embeddings and intensities 
            then we create cache (for this interval)
            """
            times = sample_times(
                event, time_last_event, num_sub, multiplier, duration, device)
            types, ratio = down_sampler.sample(cdb.event_names)
            # use .event_names not .events.keys() cuz latter has bos

            datalog.create_cache(times)

            """
            check if this is an event that we want to predict 
            in other words, if this is an event that we want to count it in the loglikelihood
            e.g., not an exogenous event like release(P)
            """
            event_term = datalog.datalog.aug_term(event['name'])
            count_its_intensity = event_term in cdb.event_names_set
            time0 = time.time()
            # added for batch-mode utilities [2021-09-21]
            if hasattr(datalog, "logger"):
                if isinstance(datalog.logger, BatchLogger):
                    datalog.logger.atom("new_event", seq_id=seq_id,
                                        event_id=i, event=event, cdb=cdb, active=active)

            if count_its_intensity and event['time'] >= cutoff_time:
                # only compute log prob for non-bos-or-eos events, AFTER cut off time
                num_event += 1.0
                num_inten += 1.0  # compute one intensity for this token
                intensity_event = datalog.compute_intensities(
                    [event['name']], cdb, active)
                inc_sum_log_intensity = torch.log(intensity_event[-1, 0] + eps)

                if checkpoint_pack is None:
                    sum_log_intensity += inc_sum_log_intensity
                else:
                    sum_log_intensity_cp += inc_sum_log_intensity

            # descript = f"Finish computing intensities: {event['name']}, {time.time() - time0}s"
            # seq_bar.set_description(descript)
            # time0 = time.time()

            compute_integral = len(types) > 0

            """
            NOTE : to speed up, we don't always compute integral for every interval 
            ideally, we should randomly skip some of them---controlled by multiplier
            if multiplier >= 1, we compute for every interval 
            otherwise, we skip some intervals
            the intervals we don't skip has total time length propto multiplier
            however, this approach requires massive revision of the current code 
            so we instead implement its asymptotic equivalence
            we instead trim the types---why they are equivalent? same in expectation!
            plan-original : fewer time points, all the types 
            plan-equivalent : same # time points, fewer types
            """
            old_num = len(types)
            if multiplier < 0.99 and old_num > 0:
                new_num = max(1, int(multiplier * old_num))
                types = random.sample(types, new_num)
                ratio *= 1.0 * old_num / new_num

            if event['name'] != 'bos' and compute_integral and event['time'] >= cutoff_time:
                # compute integral for intervals
                inc_integral, num_samples = get_inc_integral(
                    event['time'], types, datalog, cdb, active,
                    time_last_event, cutoff_time, ratio
                )
                if checkpoint_pack is None:
                    integral += inc_integral
                else:
                    integral_cp += inc_integral
                """
                track # of intensities computed 
                """
                num_inten += len(types) * num_samples  # len(types) intensities per time

            # descript = f"Finish computing integrals: {event['name']}, {time.time() - time0}s"
            # seq_bar.set_description(descript)
            # time0 = time.time()
            """
            track per-token log-probability for future evaluation 
            """
            if count_its_intensity and event['name'] != 'bos' and compute_integral and event['time'] >= cutoff_time:
                # okay to track per-token log-prob
                all_log_prob.append(
                    float(inc_sum_log_intensity) - float(inc_integral)
                )

            """
            update cell blocks
            """
            if tdb[i]['created']:
                cdb = tdb[i]['db_after_create']
                assert cdb is not None, "updated database should NOT be None"
                create_cells(
                    event, active, tdb[i]['created_cells'], cdb, device, memory_size_per_cell)

            # seq_bar.set_description(f"Finish creating cells: {event['name']}, {time.time() - time0}s")
            # time0 = time.time()

            if event['name'] != 'eos':
                """
                if anything created, cdb has been updated
                (bos problem solved)
                otherwise, use prev cdb 
                (stop(X) problem solved)
                """
                datalog.update_cells(event, cdb, active)

            # seq_bar.set_description(f"Finish updating cells: {event['name']}, {time.time() - time0}s")
            # time0 = time.time()

            if tdb[i]['killed']:  # kill cells
                kill_cells(active, dead, tdb[i]['killed_cells'])
                cdb = tdb[i]['db_after_kill']
                assert cdb is not None, "updated database should NOT be None"

            time_last_event = event['time']

            datalog.clear_cache()  # clear cache of embeddings

            """
            accumulate grads and update params if we use checkpoint training
            """
            if (i + 1) % checkpoint == 0 and event['time'] >= cutoff_time:
                if checkpoint_pack is not None:
                    # get gradients for the subseq between this and last checkpoint
                    # only valid for the segment after cutoff time
                    # print(f"i = {i}, checkpoint={checkpoint}")
                    # print(f"{int((i+1)/checkpoint)}-th checkpoint, batchsize={batchsize}")
                    loss = -(sum_log_intensity_cp - integral_cp)
                    loss.backward()
                    # print("Success")
                    del loss  # to free memory
                    if int((i + 1) / checkpoint) % batchsize == 0:
                        # update params
                        optim.step()
                        optim.zero_grad()
                    sum_log_intensity += float(sum_log_intensity_cp)
                    integral += float(integral_cp)
                    sum_log_intensity_cp, integral_cp = 0.0, 0.0  # clear segment
                detach_cells(active)  # detach history of cells

    if (i + 1) % checkpoint != 0 and event['time'] >= cutoff_time:
        if checkpoint_pack is not None:
            # finish entire seq but not update for last sugment yet
            loss = -(sum_log_intensity_cp - integral_cp)
            loss.backward()
            del loss  # to free memory
            optim.step()  # step any way
            optim.zero_grad()
            sum_log_intensity += float(sum_log_intensity_cp)
            integral += float(integral_cp)
            sum_log_intensity_cp, integral_cp = 0.0, 0.0  # clear segment
        detach_cells(active)  # detach history of cells

    log_prob = sum_log_intensity - integral
    """
    if checkpoint_pack is True, log_prob is a float 
    otherwise, it is torch tensor
    """
    return log_prob, num_event, all_log_prob, num_inten


"""
NOTE 
Chenghao mentioned that these sampled times should be sorted
but where did they get sorted?
"""


# @profile
def sample_times(
        event, time_last_event, num_sub, multiplier, duration, device):
    dtime = event['time'] - time_last_event
    assert dtime >= 0.0, f"dtime negative? {dtime}"
    num_samples = max(1, int(dtime * num_sub * multiplier / duration))

    times = torch.empty(
        size=[num_samples + 1], dtype=torch.float32, device=device
    ).uniform_(time_last_event, event['time'])
    """
    IMPORTANT: set the last to the actual time of this event 
    to facilitate the computation of its itensity and embedding
    (for log lambda and updating)
    """
    times[-1] = event['time']
    return times


"""
NOTE 
the -1-th ``sampled'' time is always the ``actual'' event time 
we should keep this in mind when we sort times and use the sorted times
"""


# @profile
def get_inc_integral(
        event_time, types, datalog, cdb, active, time_last_event, cutoff_time, ratio):
    intensities_at_times = datalog.compute_intensities(types, cdb, active)
    total_intensity = torch.sum(intensities_at_times[:-1, :])
    total_intensity *= ratio
    num_samples = intensities_at_times.size(0) - 1
    dtime = event_time - time_last_event

    frac = 1.0  # if time_last_event < cutoff_time <= event['time']
    if cutoff_time > time_last_event:
        frac *= ((event_time - cutoff_time) / (event_time - time_last_event))
    inc_integral = (total_intensity * dtime * frac) / (1.0 * num_samples)

    return inc_integral, num_samples


# @profile
def create_cells(event, active, new_cells, database, device, memory_size):
    for c in new_cells:
        lc = database.cells[c]  # logic objects of cells
        active['cells'][c] = TransformerCell(
            c, lc.dimension, lc.zeros, event['time'], device, memory_size
        )
        # active['gates'][c] = CTLSTMGate(
        #     c, lc.dimension, lc.zeros, event['time'], device
        # )
        if lc.is_event:
            active['intensity_cells'][c] = TransformerCell(
                c, 1, [], event['time'], device, memory_size
            )
            # active['intensity_gates'][c] = CTLSTMGate(
            #     c, 1, [], event['time'], device
            # )


# @profile
def kill_cells(active, dead, who_to_kill):
    for c in who_to_kill:
        dead['cells'].append(active['cells'][c])
        # dead['gates'].append(active['gates'][c])
        del active['cells'][c]
        # del active['gates'][c]
        if c in active['intensity_cells']:
            dead['intensity_cells'].append(active['intensity_cells'][c])
            # dead['intensity_gates'].append(active['intensity_gates'][c])
            del active['intensity_cells'][c]
            # del active['intensity_gates'][c]


# @profile
def detach_cells(active):
    for _, c in active['cells'].items():
        c.detach()
    for _, c in active['intensity_cells'].items():
        c.detach()


"""
NOTE 
Dec 27, 2020
already checked methods in this class, seems good 
only question is: for most of them, should they actually call old methods
next step: NeuralDatalog methods that do actual computation
things to be cautious about
(1) sorted times vs. [-1]-th actual event time
(2) how does Transformer cell do ``detach''?
"""

"""
NOTE 
prediction-related metods not updated to Transformer version yet
"""


# @profile
def get_prediction_and_gold(arguments):
    with torch.no_grad():
        # don't compute gradients 
        # save memory for eval 
        rst = get_prediction_and_gold_with_graph(arguments)
    return rst


# @profile
def get_prediction_and_gold_with_graph(arguments):
    """
    unpacked objects should match get_arguments_for_pred in manager 
    """
    seq_id, split, \
    seq, cutoff_time, \
    datalog, down_sampler, \
    thinning_sampler, \
    num_obj, \
    eps, device, verbose, memory_size_per_cell = arguments

    duration = seq[-1]['time'] + numpy.finfo(float).eps

    assert cutoff_time >= 0.0, "cutoff time must be positive"
    assert cutoff_time < duration, "we should cut off before eos"

    time_last_event = 0.0
    num_sub = len(seq) - 1  # not count BOS, i.e. # intervals
    update_cdb = False

    """
    cell blocks for this sequence 
    """
    active = {  # use dict for indexing
        'cells': dict(), 'intensity_cells': dict(),
        # 'gates': dict(), 'intensity_gates': dict()
    }
    dead = {  # use list since we don't use dead cells
        'cells': list(), 'intensity_cells': list(),
        # 'gates': list(), 'intensity_gates': list()
    }

    tdb = datalog.tdb[split][seq_id]  # temporal database
    cdb = tdb[-1]['db_after_create']  # init database

    rst = list()

    # for i, event in enumerate(tqdm(seq, desc="predicting")): # loop over seq
    for i, event in enumerate(seq):  # loop over seq
        """
        use current database to compute prediction for given history
        """
        """
        get gold next event time and type
        """
        if event['name'] != 'eos':
            """
            get next event
            """
            next_event_name = seq[i + 1]['name']
            next_event_time = seq[i + 1]['time']
        else:
            next_event_name, next_event_time = None, None

        if event['name'] != 'eos' and next_event_name != 'eos':
            """
            update database with the events except for eos
            up to 2nd last event, cuz no need to predict eos
            starting from bos
            """
            times = sample_times(
                event, time_last_event, 0, 0, duration, device)
            # use 0 to make sure we don't sample more times than needed
            datalog.create_cache(times)
            """
            update cell blocks
            """
            if tdb[i]['created']:
                cdb = tdb[i]['db_after_create']
                assert cdb is not None, "updated database should NOT be none"
                create_cells(
                    event, active, tdb[i]['created_cells'], cdb, device, memory_size_per_cell)

            if event['name'] != 'eos':
                """
                if anything created, cdb has been updated
                (bos problem solved)
                otherwise, use prev cdb 
                (stop(X) problem solved)
                """
                datalog.update_cells(event, cdb, active)

            if tdb[i]['killed']:  # kill cells
                kill_cells(active, dead, tdb[i]['killed_cells'])
                cdb = tdb[i]['db_after_kill']
                assert cdb is not None, "updated database should NOT be None"

            time_last_event = event['time']

            datalog.clear_cache()  # clear cache of embeddings

            """
            start predicting the next event after updating using this event 
            NOTE : only predict the events that we declare to have an intensity
            """
            if next_event_name is None:
                predict_next_event = False
            elif datalog.datalog.aug_term(next_event_name) not in cdb.event_names_set:
                predict_next_event = False
            else:
                predict_next_event = True

            if predict_next_event and next_event_time >= cutoff_time:
                # only predict if next event is after cut off time 
                if verbose:
                    print(f"for {seq_id}-th seq, predict after {i}-th event {event['name']} at {event['time']:.4f}")
                """
                sample time of next event
                to avoid summing all intensities 
                we down sample some event types 
                and then upweight them to approx total intensity
                """
                types, ratio = down_sampler.sample(cdb.event_names)
                # use .event_names not .events.keys() cuz latter has bos
                """
                decide what is the boundary for this time prediction
                """
                next_event_dtime = next_event_time - time_last_event
                avg_future_dtime = (duration - time_last_event) / (num_sub - i)
                look_ahead = max(next_event_dtime, avg_future_dtime)
                boundary = time_last_event + 4 * look_ahead
                # 2 times look ahead is large enough anyway---think about why!
                """
                sample possible next event times
                """
                next_event_name = datalog.datalog.aug_term(next_event_name)
                # make possible augmentation to make it datalog-friendly
                accepted_times, weights = thinning_sampler.draw_next_time(
                    [
                        time_last_event, boundary, next_event_name,
                        types, ratio, datalog, cdb, active
                    ]
                )
                """
                compute time prediction : weighted average of sampled times
                """
                time_uncond = float(torch.sum(accepted_times * weights))
                """
                compute type prediction : 
                given k(u,v)@t-ish tuple, try to predict v---the last argument
                """
                types = find_candidate_types(
                    cdb.event_names, next_event_name, datalog.datalog.extract_terms,
                    num_obj
                )
                times = sample_times(seq[i + 1], time_last_event, 0, 0, duration, device)
                # give it seq[i+1] s.t. it has actual time of next event 
                datalog.create_cache(times)
                intensities_at_times = datalog.compute_intensities(types, cdb, active)
                # size : 2 * len(types)
                datalog.clear_cache()
                intensities_at_times = intensities_at_times[-1, :]  # at actual time
                top_ids = torch.argsort(intensities_at_times, dim=0, descending=True)
                top_event_names = [types[int(top_i)] for top_i in top_ids]
                """
                compute delta time since last event 
                useful for eval metrics normalized by total variance 
                """
                dtime_uncond = time_uncond - time_last_event
                # dtime_cond = time_cond - time_last_event
                rst.append(
                    (
                        time_uncond, dtime_uncond, top_event_names,
                        next_event_time, next_event_dtime, next_event_name
                    )
                )
                if verbose:
                    print(f"our predicted time is {time_uncond:.4f} and sorted event types are :\n{top_event_names}")
                    print(
                        f"gold ({next_event_name}) ranked {top_event_names.index(next_event_name)} out of {len(top_event_names)}")
    return rst


def find_candidate_types(event_names, next_event_name, func, num_obj):
    assert next_event_name in event_names, "next event must be valid under current database"
    rst = list()
    for e in event_names:
        if match_all_but_last_n(e, next_event_name, func, num_obj):
            rst.append(e)
    return rst


def match_all_but_last_n(a, b, f, num_obj):
    # True a = k(1,2) and b = k(1,3)
    # False if a = k(1,2) and b = k(3,4)
    p_a, args_a = get_predicate_args(a, f)
    p_b, args_b = get_predicate_args(b, f)
    if p_a == p_b:
        # predicate matches 
        n = len(args_a)
        assert n == len(args_b), f"# of args not match for predicate : {p_a}"
        if n >= num_obj:
            n = num_obj
        # find the right n 
        if ','.join(args_a[:-n]) == ','.join(args_b[:-n]):
            return True
        else:
            return False
    else:
        return False


def get_predicate_args(x, f):
    i_l = x.index('(')
    i_r = x.rindex(')')
    predicate = x[:i_l]
    args = f(x[i_l + 1:i_r])  # f is a function to extract terms
    return predicate, args
