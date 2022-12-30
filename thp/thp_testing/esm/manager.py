# abstract class for both trainer and tester
# Author: Chenghao Yang
from thp.thp_testing.model.xfmr_nhp_fast import XFMRNHPFast
from thp.thp_testing.data.NHPDataset import NHPDataset, createDataLoader
from thp.thp_testing.esm.thinning import EventSampler
from thp.thp_testing.data.Dataset import prepare_dataloader
from thp.thp_training.transformer.Models import Transformer, get_non_pad_mask
from thp.thp_training.Utils import log_likelihood
from thp.thp_training.transformer.Constants import PAD
import os
import pickle
from tqdm import tqdm
import torch
import numpy
class Manager:
    def __init__(self, args):
        self.args = args
        self.train_loader, self.dev_loader, self.test_loader, num_types = prepare_dataloader(args)
        self.model = Transformer(num_types,
                                 d_model=args.d_model, n_layers=args.n_layers, d_k=args.d_k, n_head=args.n_head,
                                 d_v=args.d_v, dropout=args.dropout,
                                 d_rnn=args.d_rnn, d_inner=args.d_inner).cuda()

    def get_dataloader(self, args):
        loaders = []
        splits = ["train", 'dev', 'test']
        event_types = None
        token_types = 0
        for _split in splits:
            with open(os.path.join(args.PathDomain, f"{_split}.pkl"), "rb") as f_in:
                # latin-1 for GaTech data
                try:
                    _data = pickle.load(f_in, encoding='latin-1')
                except:
                    _data = pickle.load(f_in)
                # in general dataset used by GTech, there is no concurrent events, so set False for efficiency
                if event_types is None:
                    event_types = _data["dim_process"]
                else:
                    assert _data["dim_process"] == event_types, "inconsistent dim_process in different splits?"
                dataset = NHPDataset(_data[_split], event_types, concurrent=False, add_bos=False, add_eos=False)
                assert dataset.event_num <= event_types, f"{_split}.pkl has more event types than specified in dim_process!"
                token_types = max(token_types, dataset.num_types)
                loaders.append(createDataLoader(dataset, batch_size=args.BatchSize))
        assert token_types > event_types, f"at least we should include [PAD]! token: {token_types}, event: {event_types}"
        return loaders



    def run_one_iteration(self, model, dataLoader, mode, optimizer=None):
        assert mode in {"train", "eval"}
        if mode == "eval":
            model = model.eval()
        else:
            assert optimizer is not None
        total_log_like = 0
        num_tokens = 0
        pad_idx = PAD
        num_events = 0
        all_logs = []
        all_logs_token = []
        all_type_ll_token = []
        all_time_ll_token = []
        for batch in tqdm(dataLoader, mininterval=2, desc=f'   - ({mode}) -    ', leave=False):
            new_batch = [x.cuda() for x in batch]
            time_seq, time_delta_seq, event_seq = new_batch
            batch_non_pad_mask = get_non_pad_mask(event_seq).squeeze(-1).bool()
            enc_out, prediction = model(event_seq, time_seq)
            event_ll, non_event_ll, event_ll_normaliz = log_likelihood(model, enc_out, time_seq, event_seq)
            _batch_loss = event_ll.sum(dim=-1) - non_event_ll.sum(dim=-1)
            _loss = -torch.sum(_batch_loss)
            if mode == "train":
                _loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            total_log_like += -_loss.item()
            num_tokens += event_seq.ne(pad_idx).sum().item()
            num_events += (event_seq < pad_idx).sum().item()
            type_lls = (event_ll - event_ll_normaliz)[:, :-1]
            time_lls = (event_ll[:, 1:] - non_event_ll) - type_lls
            all_logs.extend([(x, y) for x, y in zip(_batch_loss.tolist(), event_seq.ne(pad_idx).sum(dim=-1).tolist())])
            all_logs_token.extend([(x, 1.0) for x in (event_ll[:, 1:] - non_event_ll)[batch_non_pad_mask[:, 1:]].tolist()])
            all_type_ll_token.extend([(x, 1.0) for x in type_lls[batch_non_pad_mask[:, 1:]].tolist()])
            all_time_ll_token.extend([(x, 1.0) for x in time_lls[batch_non_pad_mask[:, 1:]].tolist()])
        return total_log_like, num_tokens, num_events, all_logs, all_logs_token, all_type_ll_token, all_time_ll_token


    def create_thinningsampler(self, num_sample, num_exp):
        self.thinning_sampler = EventSampler(num_sample, num_exp)


    def run_prediction(self, model:XFMRNHPFast, dataLoader):
        self.thinning_sampler.cuda()
        results = []
        seq_id = 0
        verbose = self.args.Verbose
        thinning_sampler = self.thinning_sampler
        for _batch in tqdm(dataLoader, desc=f"   (Pred)    ", leave=False, mininterval=2):
            time_seq, time_delta_seq, event_seq = _batch
            batch_non_pad_mask = get_non_pad_mask(event_seq).squeeze(-1).bool()
            # thinning can only run in single instance mode, not in batch mode
            num_batch = time_seq.size(0)
            for i in range(num_batch):
                rst = []
                _time_seq, _event_seq = time_seq[i][batch_non_pad_mask[i]], event_seq[i][batch_non_pad_mask[i]]
                # no need to use mask here -- single instance mode
                # _attention_mask, _batch_non_pad_mask = attention_mask[i], batch_non_pad_mask[i]
                seq_len = _time_seq.size(0)
                duration = _time_seq[-1].item() + numpy.finfo(float).eps
                # in GaTech paper -- they do not predict the first event, so do we
                time_last_event = 0
                num_sub = seq_len - 1
                for j in range(seq_len - 1):
                    next_event_name, next_event_time = _event_seq[j + 1].item(), _time_seq[j + 1].item()
                    current_event_name, current_event_time = _event_seq[j].item(), _time_seq[j].item()
                    time_last_event = _time_seq[j].item()
                    if verbose:
                        print(f"for {seq_id}-th seq, predict after {j}-th event {current_event_name} at {current_event_time:.4f}")
                    next_event_dtime = next_event_time - time_last_event
                    avg_future_dtime = (duration - time_last_event) / (num_sub - j)
                    look_ahead = max(next_event_dtime, avg_future_dtime)
                    boundary = time_last_event + 4 * look_ahead
                    _event_prefix, _time_prefix = _event_seq[:j + 1].unsqueeze(0).cuda(), _time_seq[:j + 1].unsqueeze(0).cuda()
                    accepted_times, weights = thinning_sampler.draw_next_time(
                        [[_event_prefix, _time_prefix],
                        time_last_event, boundary, model]
                    )
                    time_uncond = float(torch.sum(accepted_times * weights))
                    dtime_uncond = time_uncond - time_last_event
                    intensities_at_times = model.compute_intensities_at_sampled_times(
                        _event_prefix, _time_prefix,
                        _time_seq[j + 1].reshape(1, 1).cuda()
                    )[0, 0]
                    top_ids = torch.argsort(intensities_at_times, dim=0, descending=True)
                    # since we use int to represent event names already
                    top_event_names = [int(top_i) for top_i in top_ids]
                    rst.append(
                        (
                            time_uncond, dtime_uncond, top_event_names,
                            next_event_time, next_event_dtime, next_event_name - 1
                        )
                    )
                    if verbose:
                        print(
                            f"our predicted time is {time_uncond:.4f} and sorted event types are :\n{top_event_names}")
                        print(
                            f"gold ({next_event_name}) ranked {top_event_names.index(next_event_name)} out of {len(top_event_names)}")
                results.append(rst)
                seq_id += 1
        return results




