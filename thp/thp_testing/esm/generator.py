from snhp.io.log import LogWriter
import pickle
import os
import torch
from thp.thp_testing.esm.manager import Manager
from thp.thp_testing.model.xfmr_nhp_fast import XFMRNHPFast
from thp.thp_testing.data.NHPDataset import NHPDataset, createDataLoader
from thp.thp_testing.esm.thinning import EventSampler
from thp.thp_testing.data.Dataset import prepare_dataloader
from thp.thp_training.transformer.Models import Transformer, get_non_pad_mask
from thp.thp_training.Utils import log_likelihood
from thp.thp_training.transformer.Constants import PAD
from tqdm import tqdm
import random
import math

class DummyDataset:
    def __init__(self, event_num):
        self.event_num = event_num
        self.pad_index = self.event_num
        self.num_types = event_num + 1

class Generator(Manager):

    def __init__(self, args):
        self.args = args
        if args.LoadStandardData:
            # [self.train_loader, self.dev_loader, self.test_loader] \
            #     = self.get_dataloader(args)
            # self.model = XFMRNHPFast(self.train_loader.dataset, args.ModelDim, args.Layer, args.NumHead, args.Dropout,
            #                          args.TimeEmbeddingDim).cuda()
            # self.EventNum = self.train_loader.dataset.event_num
            self.train_loader, self.dev_loader, self.test_loader, num_types = prepare_dataloader(args)
            self.EventNum = num_types
            # self.model = XFMRNHPFast(self.train_loader.dataset, args.ModelDim, args.Layer, args.NumHead, args.Dropout,
            #                          args.TimeEmbeddingDim).cuda()
            self.model = Transformer(num_types,
                                     d_model=args.ModelDim, n_layers=args.Layer, d_k=args.KeyDim, n_head=args.NumHead,
                                     d_v=args.ValueDim, dropout=args.Dropout,
                                     d_rnn=args.RNNDim, d_inner=args.InnerHiddenDim).cuda()
        else:
            self.EventNum = args.EventNum
            self.model = Transformer(self.EventNum,
                                     d_model=args.ModelDim, n_layers=args.Layer, d_k=args.KeyDim, n_head=args.NumHead,
                                     d_v=args.ValueDim, dropout=args.Dropout,
                                     d_rnn=args.RNNDim, d_inner=args.InnerHiddenDim).cuda()


        if args.LoadFromPretrain:
            self.model.load_state_dict(torch.load(args.PathModel))

        self.create_thinningsampler(args.NumSample, args.NumExp)
        os.makedirs(self.args.SavePath, exist_ok=True)
        self.log = LogWriter(os.path.join(self.args.PathModelDir, "log.txt"), vars(self.args) )
        self.log.initBest()

    def generate_seqs(self, model: XFMRNHPFast, num_seqs, min_len, max_len, max_time=10):
        self.thinning_sampler.cuda()
        results = []
        seq_id = 0
        # verbose = self.args.Verbose
        thinning_sampler = self.thinning_sampler
        bos_sets = list(range(self.EventNum))
        for i in tqdm(range(num_seqs), desc=f"   (Generate)    ", leave=False, mininterval=2):
            # time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask = _batch
            # thinning can only run in single instance mode, not in batch mode
            # num_batch = time_seq.size(0)
            # for i in range(num_batch):
            length = random.randint(min_len, max_len)
            # rst = [{"time_since_start": 0, "time_since_last_event": 0, "type_event": }]
            types = [random.sample(bos_sets, 1)[0], ]
            # _start_time = random.uniform(0, 1)
            _start_time = 0
            times = [_start_time, ]
            dtimes = [_start_time, ]
            # num_sub = length - 1
            intens = [[0] * self.args.EventNum]
            for j in range(1, length - 1):
                # _time_seq, _event_seq = time_seq[i][batch_non_pad_mask[i]], event_seq[i][batch_non_pad_mask[i]]
                _time_seq, _event_seq = torch.FloatTensor(times), torch.LongTensor(types)
                # no need to use mask here -- single instance mode
                # _attention_mask, _batch_non_pad_mask = attention_mask[i], batch_non_pad_mask[i]
                # seq_len = _time_seq.size(0)
                # do not predict second to last event, keep consistent with NDTT
                # duration = _time_seq[-1].item() + numpy.finfo(float).eps
                # [TODO]: I know it's weired -- it seems we have to implement [bos] anyway
                # but in GaTech paper -- they do not predict the first event, so do we
                # time_last_event = 0
                # time_last_event = _time_seq[0].item()
                # num_sub = seq_len - 1
                # for j in range(seq_len - 1):
                # next_event_name, next_event_time = _event_seq[j + 1].item(), _time_seq[j + 1].item()
                # current_event_name, current_event_time = _event_seq[j].item(), _time_seq[j].item()
                # time_last_event = _time_seq[j].item()
                # current_event_name, current_event_time = _event_seq[-1].item(), _time_seq[-1].item()
                time_last_event = _time_seq[-1].item()
                # if verbose:
                #     print(
                #         f"for {seq_id}-th seq, predict after {j}-th event {current_event_name} at {current_event_time:.4f}")
                # next_event_dtime = next_event_time - time_last_event
                # avg_future_dtime = (duration - time_last_event) / (num_sub - j)
                # look_ahead = max(next_event_dtime, avg_future_dtime)j
                # look_ahead = (max_time - time_last_event) / (num_sub - j)
                boundary = time_last_event + max_time
                _event_prefix, _time_prefix = _event_seq.unsqueeze(0).cuda(), _time_seq.unsqueeze(0).cuda()
                accepted_times, weights = thinning_sampler.draw_next_time(
                    [[_event_prefix, _time_prefix],
                     time_last_event, boundary, model, self.args.Patience],
                    mode="unbiased"
                )
                _time_uncond = torch.sum(accepted_times * weights)
                time_uncond = float(_time_uncond)
                dtime_uncond = time_uncond - time_last_event
                times.append(time_uncond)
                dtimes.append(dtime_uncond)
                intensities_at_times = model.compute_intensities_at_sampled_times(
                    _event_prefix, _time_prefix,
                    # _time_seq[j + 1].reshape(1, 1).cuda()
                    _time_uncond.reshape(1, 1).cuda()
                )[0, 0]
                intens.append(intensities_at_times.tolist())
                next_event_name = torch.multinomial(intensities_at_times, 1)[0].item()
                types.append(next_event_name)
                # top_ids = torch.argsort(intensities_at_times, dim=0, descending=True)
                # since we use int to represent event names already
                # top_event_names = [int(top_i) for top_i in top_ids]
                # rst.append(
                #     (
                #         time_uncond, dtime_uncond, top_event_names,
                #         next_event_time, next_event_dtime, next_event_name
                #     )
                # )
                # if verbose:
                #     print(
                #         f"our predicted time is {time_uncond:.4f} and sorted event types are :\n{top_event_names}")
                #     print(
                #         f"gold ({next_event_name}) ranked {top_event_names.index(next_event_name)} out of {len(top_event_names)}")
            rst = [
                {
                    "time_since_start": times[x],
                    "type_event": types[x],
                    "time_since_last_event": dtimes[x],
                    "intensities": intens[x]
                } for x in range(len(times))
            ]
            results.append(rst)
            seq_id += 1
        return results

    def run(self):
        splits = ["train", "test", "dev"]
        num_seqs = self.args.NumSeqs
        # train_num = int(0.8 * num_seqs)
        train_num = int(num_seqs)
        # dev_num = int(0.1 * num_seqs)
        # test_num = num_seqs - train_num - dev_num
        dev_num = 100
        test_num = 100
        torch.save(self.model.state_dict(), self.args.PathSave)
        total_tokens = 0
        for _split, _num_seqs in zip(splits, [train_num, dev_num, test_num]):
            save_path = os.path.join(self.args.SavePath, f"{_split}.pkl")
            with open(save_path, "wb") as f_out:
                data = self.generate_seqs(self.model, _num_seqs, self.args.MinLength, self.args.MaxLength,
                                          self.args.LookAheadTime)
                pickle.dump(
                    {
                        "dim_process": self.EventNum,
                        _split: data
                    }, f_out
                )
                total_tokens += sum([len(x) - 1 for x in data])
        print("impatient rate: {}".format(self.thinning_sampler.patience_counter / total_tokens))


