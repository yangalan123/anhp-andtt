from torch import nn
import torch
import math
from thp.thp_testing.model.xfmr import EncoderLayer, MultiHeadAttention, PositionwiseFeedForward

__author__ = 'Chenghao Yang, Hongyuan Mei'


# basically, this is an apple-to-apple re-implementation of GaTech THP codes in our A-NHP project code style
# have double-checked the correctness via thorough testing on their evaluation set to assure the implementation here is faithful
# in this way, we can reuse our thinning codes easily and assure correctness

class XFMRNHPFast(nn.Module):
    def __init__(self, dataset, d_model, n_layers, n_head, dropout, d_time, d_inner=128, use_norm=False):
        # d_inner only used if we want to add feedforward
        super(XFMRNHPFast, self).__init__()
        self.d_model = d_model
        self.d_time = d_time

        self.div_term = torch.exp(torch.arange(0, d_time, 2) * -(math.log(10000.0) / d_time)).reshape(1, 1, -1)
        # here num_types already includes [PAD], [BOS], [EOS]
        self.Emb = nn.Embedding(dataset.num_types, d_model, padding_idx=dataset.pad_index)
        self.layers = nn.ModuleList(
            [EncoderLayer(
                d_model + d_time,
                MultiHeadAttention(n_head, d_model + d_time, d_model, dropout, output_linear=False),
                # PositionwiseFeedForward(d_model + d_time, d_inner, dropout),
                use_residual=False,
                dropout=dropout
            )
                for _ in range(n_layers)
            ]
        )
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(d_model)
        self.inten_linear = nn.Linear(d_model, dataset.event_num)
        self.softplus = nn.Softplus()
        self.eps = torch.finfo(torch.float32).eps

    def compute_temporal_embedding(self, time):
        batch_size = time.size(0)
        seq_len = time.size(1)
        pe = torch.zeros(batch_size, seq_len, self.d_time).to(time.device)
        _time = time.unsqueeze(-1)
        div_term = self.div_term.to(time.device)
        pe[..., 0::2] = torch.sin(_time * div_term)
        pe[..., 1::2] = torch.cos(_time * div_term)
        # pe = pe * non_pad_mask.unsqueeze(-1)
        return pe

    def forward(self, event_seqs, time_seqs, batch_non_pad_mask, attention_mask, extra_times=None):
        tem_enc = self.compute_temporal_embedding(time_seqs)
        tem_enc *= batch_non_pad_mask.unsqueeze(-1)
        enc_input = torch.tanh(self.Emb(event_seqs))
        # layer_ = torch.cat([torch.zeros_like(enc_input), tem_enc], dim=-1)
        layer_ = torch.zeros_like(enc_input)
        layer_mask = (torch.eye(attention_mask.size(1)) < 1).unsqueeze(0).expand_as(attention_mask).to(
            attention_mask.device)
        if extra_times is None:
            tem_enc_layer = tem_enc
        else:
            tem_enc_layer = self.compute_temporal_embedding(extra_times)
            tem_enc_layer *= batch_non_pad_mask.unsqueeze(-1)
        # batch_size * (seq_len) * (2 * seq_len)
        _combined_mask = torch.cat([attention_mask, layer_mask], dim=-1)
        # batch_size * (2 * seq_len) * (2 * seq_len)
        _combined_mask = torch.cat([torch.ones_like(_combined_mask), _combined_mask], dim=1)
        enc_input = torch.cat([enc_input, tem_enc], dim=-1)
        _combined_non_pad_mask = torch.cat([batch_non_pad_mask, batch_non_pad_mask], dim=1)

        for enc_layer in self.layers:
            layer_ = torch.cat([layer_, tem_enc_layer], dim=-1)
            _combined_input = torch.cat([enc_input, layer_], dim=1)
            enc_output = enc_layer(
                _combined_input,
                _combined_mask
            )
            enc_output *= _combined_non_pad_mask.unsqueeze(-1)
            layer_ = torch.tanh(enc_output[:, enc_input.size(1):, :])
        if self.use_norm:
            layer_ = self.norm(layer_)

        return layer_

    def compute_loglik(self, batch):
        time_seq, time_delta_seq, event_seq, batch_non_pad_mask, attention_mask, type_mask = batch
        # 1. compute event-loglik
        enc_out = self.forward(event_seq, time_seq, batch_non_pad_mask, attention_mask)
        enc_inten = self.softplus(self.inten_linear(enc_out))
        event_lambdas = torch.sum(enc_inten * type_mask, dim=2) + self.eps
        # incase event_lambdas == 0
        event_lambdas.masked_fill_(~batch_non_pad_mask, 1.0)

        event_ll = torch.log(event_lambdas)

        # 2. compute non-event-loglik (using MC sampling to compute integral)
        num_samples = 100
        # 2.1 sample times
        # [TODO]: keep in mind the current implementation is not the right way for concurrent events
        # add [bos]@t=0
        extended_time_seq = torch.cat([torch.zeros(time_seq.size(0), 1).to(time_seq.device), time_seq], dim=-1)
        # diff_time = (time_seq[:, 1:] - time_seq[:, :-1]) * batch_non_pad_mask[:, 1:]
        diff_time = (time_seq[:, :] - extended_time_seq[:, :-1]) * batch_non_pad_mask[:, :]
        temp_time = diff_time.unsqueeze(0) * \
                    torch.rand([num_samples, *diff_time.size()], device=time_seq.device)
        # temp_time += time_seq[:, :-1].unsqueeze(0)
        temp_time += extended_time_seq[:, :-1].unsqueeze(0)
        # 2.2 compute intensities at sampled times

        # due to GPU memory limitation, we may not be able to compute all intensities at all sampled times,
        # step gives the batch size w.r.t how many sampled times we should process at each batch
        step = 20
        all_lambda = self._compute_intensities_fast(event_seq, time_seq, batch_non_pad_mask, attention_mask, temp_time,
                                                    step)
        # sum over type_events, then sum over sampled times
        all_lambda = all_lambda.sum(dim=-1)

        # 2.3 compute the empirical expectation of the summation
        all_lambda = all_lambda.sum(dim=0) / num_samples
        non_event_ll = all_lambda * diff_time

        return event_ll, non_event_ll

    def _compute_intensities_fast(self, event_seq, time_seq, batch_non_pad_mask, attention_mask, temp_time, step=20):
        # fast version, can only use in log-likelihood computation
        # assume we will sample the same number of times in each interval of the event_seqs
        all_lambda = []
        batch_size = event_seq.size(0)
        # seq_len = event_seq.size(1) - 1
        seq_len = event_seq.size(1)
        num_samples = temp_time.size(0)
        for i in range(0, num_samples, step):
            _extra_time = temp_time[i: i + step, :, :]
            _step = _extra_time.size(0)
            # _buf_time = []
            # for j in range(_step):
            #     _buf_time.append(_extra_time[:, :, j])
            _extra_time = _extra_time.reshape(_step * batch_size, -1)
            # _extra_time = _extra_time.transpose(2, 1).reshape(-1, temp_time.size(1))
            _types = event_seq.expand(_step, -1, -1).reshape(_step * batch_size, -1)
            _times = time_seq.expand(_step, -1, -1).reshape(_step * batch_size, -1)
            # _batch_non_pad_mask = batch_non_pad_mask[:, :-1].unsqueeze(0).expand(_step, -1, -1).reshape(_step * batch_size, -1)
            _batch_non_pad_mask = batch_non_pad_mask.unsqueeze(0).expand(_step, -1, -1).reshape(_step * batch_size, -1)
            # _attn_mask = attention_mask[:, :-1, :-1].unsqueeze(0).expand(_step, -1, -1, -1).reshape(_step * batch_size, seq_len, seq_len)
            _attn_mask = attention_mask.unsqueeze(0).expand(_step, -1, -1, -1).reshape(_step * batch_size, seq_len,
                                                                                       seq_len)
            # _times = torch.cat([time] * _extra_time.size(2), dim=0)

            # _enc_output = self.forward(_types[:, :-1], _times[:, :-1],
            _enc_output = self.forward(_types, _times,
                                       _batch_non_pad_mask,
                                       _attn_mask,
                                       _extra_time)
            all_lambda.append(self.softplus(self.inten_linear(_enc_output)).reshape(_step, batch_size, seq_len, -1))
        all_lambda = torch.cat(all_lambda, dim=0)
        return all_lambda

    def compute_intensities_at_sampled_times(self, event_seq, time_seq, sampled_times):
        # Assumption: all the sampled times are distributed [time_seq[...,-1], next_event_time]
        # used for thinning algorithm
        num_batches = event_seq.size(0)
        seq_len = event_seq.size(1)
        assert num_batches == 1, "Currently, no support for batch mode (what is a good way to do batching in thinning?)"
        if num_batches == 1 and num_batches < sampled_times.size(0):
            _sample_size = sampled_times.size(0)
            # multiple sampled_times
            event_seq = event_seq.unsqueeze(0).expand(_sample_size, num_batches, seq_len).reshape(_sample_size, seq_len)
            time_seq = time_seq.unsqueeze(0).expand(_sample_size, num_batches, seq_len).reshape(_sample_size, seq_len)
        assert (time_seq[:, -1:] <= sampled_times).all(), "sampled times must occur not earlier than last events!"
        num_samples = sampled_times.size(1)

        # 1. prepare input embeddings for "history"
        tem_enc = self.compute_temporal_embedding(time_seq)
        enc_input = torch.tanh(self.Emb(event_seq))
        layer_ = torch.zeros((sampled_times.size(0), sampled_times.size(1), enc_input.size(-1))).to(
            sampled_times.device)
        enc_input = torch.cat([enc_input, tem_enc], dim=-1)
        tem_layer_ = self.compute_temporal_embedding(sampled_times)

        # 2. prepare attention mask
        attention_mask = torch.ones((num_batches, seq_len + num_samples, seq_len + num_samples)).to(event_seq.device)
        # by default, regard all_sampled times to be equal to the last_event_time
        # recall that we use 1 for "not attending", 0 for "attending"
        # t_i == sampled_t
        attention_mask[:, seq_len:, :seq_len - 1] = 0
        # t_i < sampled_t
        attention_mask[:, seq_len:, seq_len - 1][time_seq[:, -1:] < sampled_times] = 0
        for enc_layer in self.layers:
            layer_ = torch.cat([layer_, tem_layer_], dim=-1)
            _combined_input = torch.cat([enc_input, layer_], dim=1)
            enc_output = enc_layer(
                _combined_input,
                attention_mask
            )
            layer_ = torch.tanh(enc_output[:, enc_input.size(1):, :])
        if self.use_norm:
            layer_ = self.norm(layer_)

        sampled_intensities = self.softplus(self.inten_linear(layer_))

        return sampled_intensities
