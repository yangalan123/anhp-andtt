import numpy as np
import time
import torch
from snhp.logic.neuraldatalog import DatalogPlus
from snhp.logic.neuraldatalog import NeuralDatalog
from snhp.neural.aggr import Aggregation
from snhp.neural.linear import LinearZero
from torch.nn import Tanh, Softplus, Linear, ModuleList

from andtt.logic.xfmr_db import XMFRDatabaseFast as Database
from andtt.neural.attention import ScaledDotAttention
from andtt.neural.cell import TransformerCell


class XFMRNeuralDatalog(NeuralDatalog):
    def __init__(self, args, device=None):

        print("init neural datalog database")
        self.datalog = DatalogPlus()
        self.args = args
        te_mode = args.TimeEmbeddingMode
        te_embed_dim = args.TimeEmbeddingDim
        layer = args.Layer
        self.tdb = dict()  # tdb stands for temporal database
        device = device or 'cpu'
        if args.UseGPU:
            device = "cuda"
        self.device = torch.device(device)
        assert te_mode in ["Linear", "Sine"], f"Unknown time embedding mode : {te_mode}"
        self.te_mode = te_mode
        # when te_embed_dim=0, we will use masking specified in self.update_see_params_per_idx_and_dim()
        # so in order to avoid weired behavior, we will use 1 when te_embed_dim=0
        self.te_embed_dim = max(te_embed_dim, 1)
        assert layer >= 1 and type(layer) == int, "The number of layer must be a positive integer."
        self.layer = layer
        # only two modes are supported: extra_dim and extra_layer
        self.intenmode = self.args.IntensityComputationMode
        assert self.intenmode in ["extra_dim", "extra_layer"]

    def init_params(self):
        # Keep deductive part as the same as ICML 2020
        self.transform_param = dict()
        self.transform_param_intensity = dict()  # for extra-dimension of intensity

        self.see_param_key = dict()
        self.see_param_query = dict()
        self.see_param_value = dict()

        # add "self-att" to deal with the situation when there is only one event to attend to
        self.see_param_self_att_key = dict()
        self.see_param_self_att_query = dict()
        self.see_param_self_att_value = dict()

        self.see_param_intensity_key = dict()  # for extra-dimension of intensity
        self.see_param_intensity_query = dict()  # for extra-dimension of intensity
        self.see_param_intensity_value = dict()  # for extra-dimension of intensity


        # add "self-att" to deal with the situation when there is only one event to attend to
        self.see_param_intensity_self_att_key = dict()
        self.see_param_intensity_self_att_query = dict()
        self.see_param_intensity_self_att_value = dict()

        if self.intenmode == "extra_layer":
            self.intensity_layer = dict()

        self._initialize_time_embedding()

        self.aggregation = dict()
        self.activation = dict()
        self.activation['tanh'] = Tanh()
        self.activation['softplus'] = Softplus()
        self.activation["attention"] = ScaledDotAttention()

        self.transform_idx_and_dim = set()
        self.see_idx_and_dim = set()

    def map_params(self):
        full_dict = {
            'transform_param': self.transform_param,
            'transform_param_intensity': self.transform_param_intensity,
            'see_param_key': self.see_param_key,
            'see_param_value': self.see_param_value,
            'see_param_query': self.see_param_query,
            'see_param_intensity_key': self.see_param_intensity_key,
            'see_param_intensity_value': self.see_param_intensity_value,
            'see_param_intensity_query': self.see_param_intensity_query,
            'see_param_self_att_key': self.see_param_self_att_key,
            'see_param_self_att_value': self.see_param_self_att_value,
            'see_param_self_att_query': self.see_param_self_att_query,
            'see_param_intensity_self_att_key': self.see_param_intensity_self_att_key,
            'see_param_intensity_self_att_value': self.see_param_intensity_self_att_value,
            'see_param_intensity_self_att_query': self.see_param_intensity_self_att_query,
            'aggregation': self.aggregation,
        }
        if self.intenmode == "extra_layer":
            full_dict = {
                'transform_param': self.transform_param,
                'see_param_key': self.see_param_key,
                'see_param_value': self.see_param_value,
                'see_param_query': self.see_param_query,
                'see_param_self_att_key': self.see_param_self_att_key,
                'see_param_self_att_value': self.see_param_self_att_value,
                'see_param_self_att_query': self.see_param_self_att_query,
                'aggregation': self.aggregation,
                "intensity_layer": self.intensity_layer,
            }
        return full_dict

    def count_params(self):
        cnt = 0
        for _, param_dict in self.map_params().items():
            for _, param in param_dict.items():
                if isinstance(param, ModuleList):
                    for i in range(len(param)):
                        cnt += param[i].count_params()
                else:
                    cnt += param.count_params()
        return cnt


    def _initialize_time_embedding(self):
        if self.te_mode == "Linear":
            self.te_embed = Linear(1, self.te_embed_dim).to(self.device)
        elif self.te_mode == "Sine":
            # bias : 0 or pi/2
            # recall sin (x + pi/2) = cos (x) -> sin (wx + pi/2 ) = cos (wx)
            # here w is the "weight" and {0, pi/2}^{D_time} is the "bias"
            self.te_embed_bias = torch.zeros(self.te_embed_dim, requires_grad=False, device=self.device)
            self.te_embed_bias[::2] = np.pi / 2

            self.te_embed_weight = [1 / np.power(10000, (x - x % 2) / self.te_embed_dim) for x in range(self.te_embed_dim)]
            self.te_embed_weight = torch.Tensor(self.te_embed_weight).to(self.device)

    def get_time_embedding(self, time):
        if self.te_mode == "Linear":
            ret = self.te_embed(time)
            if ret.dim() == 1:
                return ret.unsqueeze(0)
            else:
                return ret
        elif self.te_mode == "Sine":
            if isinstance(time, float):
                return torch.sin(time * self.te_embed_weight + self.te_embed_bias)
            else:
                assert isinstance(time, list) or isinstance(time, torch.Tensor)
                if not (isinstance(time, torch.Tensor) and time.dim() == 2 and time.size(1) == 1):
                    time = torch.Tensor(time).reshape(-1, 1)
                return torch.sin(time.float().to(self.device).matmul(self.te_embed_weight.reshape(1, -1)) + self.te_embed_bias)
        else:
            raise NotImplementedError("Currently, time embedding only support Linear and Sine mode")

    def get_current_database(self):
        return Database(self.datalog)

    def create_cache(self, times):
        """
        cache the computed embeddings
        cache is times-specific (i.e. interval-specific)
        """
        assert isinstance(times, torch.Tensor), "it has to be a tensor"
        # add extra sorting to help improve the efficiency
        self.cache_times = times.sort()[0]
        # +1 : for 0-th layer
        self.cache_emb = {
            'true(specialaug)': torch.stack(
                [torch.zeros(*times.size(), 1, device=self.device) for _ in range(self.layer + 1)], dim=0),
            'bos(specialaug)': torch.stack(
                [torch.zeros(*times.size(), 1, device=self.device) for _ in range(self.layer + 1)], dim=0)
        }
        self.cache_inten = {}  # cache intensities

    def create_intensity_layer(self, functor, dim):
        if functor not in self.intensity_layer:
            # why no need to specify in_zero or output_zero here?
            # because we already apply cell_mask after embedding computation!
            # since intensity layer is built upon the embedding
            # no need to specifiy any in/out_zero
            self.intensity_layer[functor] = LinearZero(functor, dim, 1, [], [], device=self.device).to(self.device)

    def update_transform_params_per_idx_and_dim(self, idx_and_dim):
        try:
            betaparam_i, mparam_i, in_dim, in_z, out_dim, out_z, head_functor, head_is_event = idx_and_dim
        except:
            # for backward compatibility
            betaparam_i, mparam_i, in_dim, in_z, out_dim, out_z, head_is_event = idx_and_dim
        if betaparam_i not in self.aggregation:
            self.aggregation[betaparam_i] = Aggregation(betaparam_i, device=self.device).to(self.device)
        if mparam_i not in self.transform_param:
            self.transform_param[mparam_i] = LinearZero(
                mparam_i, in_dim, out_dim, in_z, out_z, device=self.device).to(self.device)
        if head_is_event and mparam_i not in self.transform_param_intensity:
            self.transform_param_intensity[mparam_i] = LinearZero(
                mparam_i, in_dim, 1, in_z, [], device=self.device).to(self.device)
        # in case some events can only be proved via deductive rules
        if head_is_event and self.intenmode == "extra_layer":
            self.create_intensity_layer(head_functor, out_dim)


    def cuda(self):
        for param in self.chain_params():
            param.cuda()
        pass

    def update_see_params_per_idx_and_dim(self, idx_and_dim):
        """
        NOTE
        all dimensions such as cell_dim & in_dim are already modified by DatalogPlus
        e.g., if input is concat of 2 0-dim vectors
        then in_dim = 2, not 0
        """
        betaparam_i, mparam_i, in_dim, in_z, cell_dim, cell_z, event_dim, head_functor, head_is_event = idx_and_dim
        if betaparam_i not in self.aggregation:
            self.aggregation[betaparam_i] = Aggregation(betaparam_i, device=self.device).to(self.device)
        """
        NOTE
        how te_embed_dim affect args of LinearZero
        if te_embed_dim = 0, we should input 
        in_dim + 1 as in_dim
        in_z + [in_dim] as in_z
        NOTE 
        te_embed should ALWAYS be concat at the end, to not mess in_z up
        """
        if self.te_embed_dim == 0:
            key_in_dim, val_in_dim = in_dim + 1, in_dim + 1
            key_in_z, val_in_z = in_z + (in_dim, ), in_z + (in_dim, )
            query_in_dim = cell_dim + 1
            query_in_z = cell_z + (cell_dim, )
        elif self.te_embed_dim > 0:
            key_in_dim, val_in_dim = in_dim + self.te_embed_dim, in_dim + self.te_embed_dim
            key_in_z, val_in_z = in_z, in_z
            query_in_dim = cell_dim + self.te_embed_dim
            query_in_z = cell_z
        else:
            raise Exception(f"Invalid te_embed_dim : {self.te_embed_dim}")

        if mparam_i not in self.see_param_key:
            # Key = W^K[ [[e]](s); [[c_1]](s); ... ; [[c_n]](s); te(s) ]
            # Key does not depend on current t -- since we are attending to a past instantiation
            self.see_param_key[mparam_i] = ModuleList([LinearZero(
                mparam_i, key_in_dim, cell_dim, key_in_z, cell_z, device=self.device) for _ in range(self.layer)]).to(self.device)
            # Val = W^V[ [[e]](s); [[c_1]](s); ... ; [[c_n]](s); te(s) ]
            self.see_param_value[mparam_i] = ModuleList([LinearZero(
                mparam_i, val_in_dim, cell_dim, val_in_z, cell_z, device=self.device) for _ in range(self.layer)]).to(self.device)
            # Que = W^Q[ [[h]](s^+); te(t) ]
            self.see_param_query[mparam_i] = ModuleList([LinearZero(
                mparam_i, query_in_dim, cell_dim, query_in_z, cell_z, device=self.device) for _ in range(self.layer)]).to(self.device)

        if head_is_event and mparam_i not in self.see_param_intensity_key:
            self.see_param_intensity_key[mparam_i] = ModuleList([LinearZero(
                mparam_i, key_in_dim, 1, key_in_z, [], device=self.device) for _ in range(self.layer)]).to(self.device)

            self.see_param_intensity_value[mparam_i] = ModuleList([LinearZero(
                mparam_i, val_in_dim, 1, val_in_z, [], device=self.device) for _ in range(self.layer)]).to(self.device)

            self.see_param_intensity_query[mparam_i] = ModuleList([LinearZero(
                mparam_i, query_in_dim, 1, query_in_z, [], device=self.device) for _ in range(self.layer)]).to(self.device)

        if head_functor not in self.see_param_self_att_key:
            self.see_param_self_att_key[head_functor] = ModuleList([LinearZero(
                head_functor, query_in_dim, cell_dim, query_in_z, cell_z, device=self.device) for _ in range(self.layer)]).to(self.device)
            self.see_param_self_att_value[head_functor] = ModuleList([LinearZero(
                head_functor, query_in_dim, cell_dim, query_in_z, cell_z, device=self.device) for _ in range(self.layer)]).to(self.device)
            self.see_param_self_att_query[head_functor] = ModuleList([LinearZero(
                head_functor, query_in_dim, cell_dim, query_in_z, cell_z, device=self.device) for _ in range(self.layer)]).to(self.device)

        if head_is_event and head_functor not in self.see_param_intensity_self_att_key:
            self.see_param_intensity_self_att_key[head_functor] = ModuleList([LinearZero(
                head_functor, query_in_dim, 1, query_in_z, [], device=self.device) for _ in range(self.layer)]).to(self.device)
            self.see_param_intensity_self_att_value[head_functor] = ModuleList([LinearZero(
                head_functor, query_in_dim, 1, query_in_z, [], device=self.device) for _ in range(self.layer)]).to(self.device)
            self.see_param_intensity_self_att_query[head_functor] = ModuleList([LinearZero(
                head_functor, query_in_dim, 1, query_in_z, [], device=self.device) for _ in range(self.layer)]).to(self.device)

        # in case some events can only be proved via update rules
        if head_is_event and self.intenmode == "extra_layer":
            self.create_intensity_layer(head_functor, cell_dim)


    def _concat_time_for_key_and_value(self, embedding, time):
        # utility function
        # only two possible situations: 1-to-1 and M-to-M
        # embedding: 1) list or Tensor of shape [dim,] (only in 1-to-1)
        #  or 2) Tensor of shape [len(time), dim] (only in M-to-M)
        # Key = W^K[[[e]](s); [[c_1]](s); ... ; [[c_n]](s); te(s)]
        if isinstance(time, float):
            # 1-to-1
            if isinstance(embedding, list):
                return torch.Tensor(embedding + self.get_time_embedding(time).tolist())
            elif isinstance(embedding, torch.Tensor):
                assert embedding.dim() == 1
                return torch.cat([embedding, self.get_time_embedding(time), ], dim=0)
        else:
            # M-to-M
            assert isinstance(time, list) or isinstance(time, torch.Tensor), \
                f"Unsupported parameter types for param 'time': {time}: {type(time)}"
            assert isinstance(embedding, torch.Tensor), \
                f"Unsupported parameter types for param 'embedding': {embedding}: {type(embedding)}"
            assert embedding.dim() == 2 and embedding.size(0) == len(time), \
                f"Please check the shape of param 'embedding': {embedding.size()}"
            times = torch.tensor(time).reshape(-1, 1)
            return torch.cat([embedding, self.get_time_embedding(times)], dim=-1)

    def _concate_time_for_query(self, embedding, time):
        if isinstance(time, float):
            # 1-to-1
            if isinstance(embedding, list):
                return torch.cat([torch.Tensor(embedding), self.get_time_embedding(time)], dim=-1)
            elif isinstance(embedding, torch.Tensor):
                assert embedding.dim() == 1
                return torch.cat([embedding, self.get_time_embedding(time)], dim=-1)
            else:
                raise NotImplementedError(
                    f"Unsupported parameter types for param 'embedding': {embedding}: {type(embedding)}"
                )
        else:
            # M-to-M
            assert isinstance(time, list) or isinstance(time, torch.Tensor), \
                f"Unsupported parameter types for param 'time': {time}: {type(time)}"
            assert isinstance(embedding, torch.Tensor), \
                f"Unsupported parameter types for param 'embedding': {embedding}: {type(embedding)}"
            assert embedding.dim() == 2 and embedding.size(0) == len(time), \
                f"Please check the shape of param 'embedding': {embedding.size()}"
            times = torch.tensor(time).reshape(-1, 1)
            return torch.cat([embedding, self.get_time_embedding(times)], dim=-1)

    def judge_batch_mode(self, times):
        if torch.is_tensor(times) and len(times.shape) > 1:
            return True
        return False

    def compute_intensities(self, event_types, cdb, active):
        """
        compute intensities for event types at times
        using the current database, i.e. cdb
        """
        for i, e in enumerate(event_types):
            if self.intenmode == "extra_dim":
                if e not in self.cache_inten:
                    self.compute_intensity_(e, cdb, active)
            else:
                # in "extra_layer" mode, intensities are directly computed on embeddings
                if e not in self.cache_inten:
                    if e not in self.cache_emb:
                        self.compute_embedding_(e, cdb, active)
                    functor = self.datalog.get_functor(e)
                    self.cache_inten[e] = self.activation["softplus"](self.intensity_layer[functor](self.cache_emb[e]))
        # only use the last layer results
        intensities = [self.cache_inten[e][-1].squeeze(-1) for e in event_types]
        return torch.stack(intensities, dim=-1)

    def compute_intensity_bounds(self, event_types, cdb, active):
        """
        compute intensity bounds for event types at times
        using the current database, i.e. cdb
        """
        self.cache_inten_bound = dict()
        for i, e in enumerate(event_types):
            if e not in self.cache_inten_bound:
                self.compute_intensity_bound_(e, cdb, active)
        # only use the last layer results
        intensity_bounds = [self.cache_inten_bound[e][-1].squeeze(-1) for e in event_types]
        del self.cache_inten_bound
        return torch.stack(intensity_bounds, dim=-1)  # 1 * len(event_types)

    def compute_embeddings(self, terms, cdb, active):
        # cdb for current database
        for i, t in enumerate(terms):
            if t not in self.cache_emb:
                self.compute_embedding_(t, cdb, active)
        # only use the last layer
        embs = [self.cache_emb[t][-1] for t in terms]
        # can't stack embs cuz they may have different dimensions
        return embs

    def compute_embedding_(self, term, cdb, active, cache=True, times=None):
        provable = term in cdb.terms
        hascell = term in cdb.cells
        assert provable or hascell, \
            f"why compute embedding of {term} if not provable and no cell?"
        deductive_embedding_collection = []
        FLAG_MULTI_SAMPLE = False
        if times is None:
            FLAG_MULTI_SAMPLE = self.judge_batch_mode(self.cache_times)
        if provable:
            t = cdb.terms[term]
            for r_i, b_facts_i in t.transform_edges.items():
                for b_i, facts_i in b_facts_i.items():
                    to_aggr = []
                    for body_i, m_i in facts_i:
                        to_cat = []
                        for subgoal in body_i:
                            if subgoal not in self.cache_emb:
                                self.compute_embedding_(subgoal, cdb, active)
                            to_cat.append(self.cache_emb[subgoal])

                        output = self.transform_param[m_i](torch.cat(to_cat, dim=-1))
                        to_aggr.append(output)
                    aggred = self.aggregation[b_i](torch.stack(to_aggr, dim=0))
                    # after aggregation, sum them up
                    # r_i and b_i is 1-to-1
                    deductive_embedding_collection.append(aggred)
        if len(deductive_embedding_collection) > 0:  # if embedding_output: # maybe quicker
            deductive_embedding = torch.stack(deductive_embedding_collection, dim=0).sum(dim=0)
        else:
            deductive_embedding = 0
        # just a placeholder so do can use the same formula at the end of this function
        cell_embedding = 0
        if hascell:  # if it is an active Transformer Cell
            term_cell = active["cells"][term]
            if len(deductive_embedding_collection) > 0:
                # layer 0: tanh(0 + :-) = tanh(:-)
                # full_embedding = self.activation["tanh"](deductive_embedding)
                full_embedding = self.activation["tanh"](deductive_embedding[0])
            else:
                # layer 0: tanh(0 + none) = tanh(0)
                if times is None:
                    full_embedding = torch.zeros((*self.cache_times.size(), term_cell.cell_dim), device=self.device)
                else:
                    full_embedding = torch.zeros((*times.size(), term_cell.cell_dim), device=self.device)

            full_embed_collection = [full_embedding, ]

            for _layer in range(self.layer):
                if times is None:
                    cell_embedding = self.compute_block_embedding_via_att(term_cell, self.cache_times, full_embedding,
                                                                          _layer)
                else:
                    cell_embedding = self.compute_block_embedding_via_att(term_cell, times, full_embedding, _layer)
                _deductive_embedding = 0
                if len(deductive_embedding_collection) > 0:
                    _deductive_embedding = deductive_embedding[_layer]
                full_embedding = self.activation['tanh'](_deductive_embedding + cell_embedding)
                full_embed_collection.append(full_embedding)
            if cache:
                self.cache_emb[term] = torch.stack(full_embed_collection, dim=0)
            else:
                return full_embed_collection
        else:
            if not FLAG_MULTI_SAMPLE:
                full_embedding = self.activation["tanh"](deductive_embedding).expand((self.layer + 1), -1, -1)
            else:
                full_embedding = self.activation["tanh"](deductive_embedding).expand((self.layer + 1),
                                                                                     *self.cache_times.size(), -1)
            if cache:
                self.cache_emb[term] = full_embedding
            else:
                return full_embedding

    def compute_block_embedding_via_att(self, cell: TransformerCell, times,
                                        ### [New for Multi-layer]
                                        full_embedding_last_layer, layer,
                                        for_intensity_dim=False):
        start_time = time.time()
        # for searching [[h]](s^+)
        # for detailed explanations about "left" and "right", see models.cell.TransformerCell.search_most_recent_until_t
        # most_recent_idxs_h = list(cell.search_most_recent_until_t(times, "left"))
        # no need to worry about x < 0 here
        # since we have done checking in models.cell.TransformerCell.search_most_recent_until_t
        # most_recent_states = [cell.history[x]["state"] for x in most_recent_idxs_h]

        # MULTI_SAMPLE_SUPPORT: only used when doing prediction using thinning
        # multiple samples support during prediction
        FLAG_MULTI_SAMPLE = self.judge_batch_mode(times)
        times_shape, times_recover_index = times.shape, None
        if not FLAG_MULTI_SAMPLE:
            most_recent_idxs, truncated_history = list(cell.search_most_recent_until_t(times, "left"))
        else:
            # times_sort_index : unsorted batched data -> sorted flattened data
            # times_recover_index : assume the times_sort_index is [0, 2, 1], but the original order should be [0, 1, 2]
            # how to recover [0, 1, 2] from [0, 2, 1]? Give torch [0, 2, 1].argsort()!
            # i.e., a=torch.Tensor([0, 1, 2]), b = torch.Tensor([0,2,1]), c=b.argsort(),
            # b[c] === a
            times, times_sort_index = times.reshape(-1).sort()
            times_recover_index = times_sort_index.argsort()
            most_recent_idxs, truncated_history = list(cell.search_most_recent_until_t(times, "left"))
            if len(full_embedding_last_layer.shape) > 2:
                full_embedding_last_layer = full_embedding_last_layer.reshape(len(times), -1)
                # sorting according to the chronical order
                full_embedding_last_layer = full_embedding_last_layer[times_sort_index]

        ret = []
        if not hasattr(cell, "functor"):
            cell_functor = self.datalog.get_functor(cell.name)
            cell.functor = cell_functor
        else:
            cell_functor = cell.functor
        if not for_intensity_dim:
            query_fn = self.see_param_query
            self_key_fn = self.see_param_self_att_key[cell_functor][layer]
            self_value_fn = self.see_param_self_att_value[cell_functor][layer]
            self_query_fn = self.see_param_self_att_query[cell_functor][layer]
        else:
            query_fn = self.see_param_intensity_query
            self_key_fn = self.see_param_intensity_self_att_key[cell_functor][layer]
            self_value_fn = self.see_param_intensity_self_att_value[cell_functor][layer]
            self_query_fn = self.see_param_intensity_self_att_query[cell_functor][layer]

        self_values = self_value_fn(self._concate_time_for_query(full_embedding_last_layer, times)).reshape(len(times), -1)
        if len(cell.history) == 1 or max([x["history_id"] for x in most_recent_idxs]) == 0:
            # declared fact, like "tag(action)."
            # or simply nothing can attend (just at the time that the head is created by the update rule)
            return (self_values * cell.cell_mask).reshape(list(times_shape) + [-1])
        keys = torch.stack([x["Key"][layer] for x in truncated_history[1:]], dim=0)
        values = torch.stack([x["Value"][layer] for x in truncated_history[1:]], dim=0)
        
        self_keys = self_key_fn(self._concate_time_for_query(full_embedding_last_layer, times)).reshape(len(times), 1, -1)
        self_querys = self_query_fn(self._concate_time_for_query(full_embedding_last_layer, times)).reshape(len(times), 1, -1)

        mparam2key = dict()
        mparam2value = dict()
        history_counter = 1
        time_counter = 0
        for blocks in most_recent_idxs:
            history_id, _times = blocks["history_id"], blocks["times"]
            if history_id == 0:
                # non-zero probability sample the exact time when the first update event happened
                # at that time, that event still does not have influence, so we return zero-vectors
                ret.append(torch.stack([truncated_history[0]["state"] for _ in range(len(_times))], dim=0))
                time_counter += len(_times)
                continue
            assert history_id + 1 >= history_counter
            # O(KN) -> O(N): use last_counter to complete incremental KQV collection
            for record_i in range(history_counter, history_id + 1):
                mparam_i = truncated_history[record_i]["mparam_i"]
                if mparam_i not in mparam2key:
                    mparam2key[mparam_i] = []
                    mparam2value[mparam_i] = []
                mparam2key[mparam_i].append(keys[record_i - 1])
                mparam2value[mparam_i].append(values[record_i - 1])
            history_counter = history_id + 1
            # add-zero self-loop
            normalization_constant = 1
            _summed_vals = []
            for mparam_i in mparam2key:
                # each rule will have its own attention head
                _query = self._concate_time_for_query(
                    full_embedding_last_layer[time_counter: time_counter + len(_times)], _times)
                _qs = query_fn[mparam_i][layer](_query).reshape(len(_times), 1, -1).expand(len(_times),
                                                                                           len(mparam2key[mparam_i]),
                                                                                           -1)
                _ks = torch.stack(mparam2key[mparam_i], dim=0).reshape(1, len(mparam2key[mparam_i]), -1)\
                    .expand(len(_times), -1, -1)
                _vs = torch.stack(mparam2value[mparam_i], dim=0).reshape(len(mparam2value[mparam_i]), -1)
                _qs = torch.cat([_qs, self_querys[time_counter: time_counter + len(_times)] ], dim=1)
                _ks = torch.cat([_ks, self_keys[time_counter: time_counter + len(_times)] ], dim=1)
                # now normalizing will conduct based on mparam_i, rather than do cross-rule normalization
                # [Rule-specific normalization]
                _att_weight = torch.softmax(
                    torch.sum(
                        _qs * _ks
                        / self.args.AttentionTemperature
                        ,
                        dim=-1
                    ), dim=1
                )
                _summed_vals.append(
                    _att_weight[:, :len(mparam2value[mparam_i])].matmul(
                        _vs
                    )
                    + _att_weight[:, len(mparam2value[mparam_i]):] * self_values[
                                                                       time_counter: time_counter + len(_times)]
                )
            ret.append(torch.sum(torch.stack(_summed_vals, dim=1), dim=1) / normalization_constant * cell.cell_mask)

            time_counter += len(_times)

        if not FLAG_MULTI_SAMPLE:
            return torch.cat(ret, dim=0)
        else:
            assert times_recover_index is not None
            permuted_ret = torch.cat(ret, dim=0)
            ret = permuted_ret[times_recover_index].reshape(list(times_shape) + [-1])
            return ret

    def compute_intensity_(self, event_type, cdb, active, cache=True, times=None):
        """
        compute intensity of a given event type
        """
        event_type = self.datalog.aug_term(event_type)
        provable = event_type in cdb.terms
        hascell = event_type in cdb.cells
        assert provable, \
            f"event type {event_type} not provable? no way cuz it must be head of :- rule"
        FLAG_MULTI_SAMPLE = False
        if times is None:
            FLAG_MULTI_SAMPLE = self.judge_batch_mode(self.cache_times)
        deductive_intensity_collection = []
        t = cdb.terms[event_type]
        for r_i, b_facts_i in t.transform_edges.items():
            for b_i, facts_i in b_facts_i.items():
                to_aggr = []
                for body_i, m_i in facts_i:
                    to_cat = []
                    for subgoal in body_i:
                        if subgoal not in self.cache_emb:
                            self.compute_embedding_(subgoal, cdb, active)
                        to_cat.append(self.cache_emb[subgoal][-1])
                        # only take the top most layer (achieved by ``[-1]'')
                        # when we compute the deductive parts.
                    output = self.transform_param_intensity[m_i](torch.cat(to_cat, dim=-1))
                    to_aggr.append(output)
                aggred = self.aggregation[b_i](torch.stack(to_aggr, dim=0))
                # after aggregation, sum them up
                # r_i and b_i is 1-to-1
                deductive_intensity_collection.append(aggred)
        if len(deductive_intensity_collection) > 0:  # ``if intensity_output: '' works and maybe quicker
            deductive_intensity = torch.stack(deductive_intensity_collection, dim=0).sum(dim=0)
        else:
            deductive_intensity = 0
        cell_intensity = 0.0
        if hascell:  # if it is an active Transformer Cell
            term_cell = active["intensity_cells"][event_type]
            if len(deductive_intensity_collection) > 0:
                # layer 0: tanh(0 + :-) = tanh(:-)
                full_intensity = self.activation["softplus"](deductive_intensity)
            else:
                # layer 0: tanh(0 + none) = tanh(0)
                if times is None:
                    full_intensity = torch.zeros((*self.cache_times.size(), term_cell.cell_dim), device=self.device)
                else:
                    full_intensity = torch.zeros((*times.size(), term_cell.cell_dim), device=self.device)

            full_inten_collection = [full_intensity, ]

            for _layer in range(self.layer):
                if times is None:
                    cell_intensity = self.compute_block_embedding_via_att(term_cell, self.cache_times, full_intensity,
                                                                          _layer, for_intensity_dim=True)
                else:
                    cell_intensity = self.compute_block_embedding_via_att(term_cell, times, full_intensity, _layer,
                                                                          for_intensity_dim=True)
                full_intensity = self.activation['softplus'](deductive_intensity + cell_intensity)
                full_inten_collection.append(full_intensity.squeeze(-1))
            if cache:
                self.cache_inten[event_type] = torch.stack(full_inten_collection, dim=0)
            else:
                return full_inten_collection
        else:
            # actually it is 0-th layer embedding
            # +1 for 0-th layer
            if not FLAG_MULTI_SAMPLE:
                full_intensity = self.activation["softplus"](deductive_intensity).expand((self.layer + 1), -1, -1)
            else:
                full_intensity = self.activation["softplus"](deductive_intensity).expand((self.layer + 1),
                                                                                         *self.cache_times.size(), -1)
            if cache:
                self.cache_inten[event_type] = full_intensity
            else:
                return full_intensity


    def update_cells(self, event, cdb, active):
        event_type = event['name']
        assert event_type is not 'eos', "EOS doesn't update cells"
        event_type = self.datalog.aug_term(event_type)
        if event_type in cdb.events:
            for c in cdb.events[event_type].who_see_it:
                # c is head of <- rule with event_type as condition0
                self.compute_new_cell(
                    c, event_type, cdb, active, event['time'], False)
                if cdb.cells[c].is_event:
                    self.compute_new_cell(
                        c, event_type, cdb, active, event['time'], True)

    def compute_intensity_bound_(self, event_type, cdb, active):
        """
        compute intensity bound of a given event type
        no need to compute embeddings
        """
        event_type = self.datalog.aug_term(event_type)
        provable = event_type in cdb.terms
        hascell = event_type in cdb.cells
        assert provable, \
            f"event type {event_type} not provable? no way cuz it must be head of :- rule"
        intensity = 0.0
        t = cdb.terms[event_type]
        for r_i, b_facts_i in t.transform_edges.items():
            for b_i, facts_i in b_facts_i.items():
                to_aggr = []
                for body_i, m_i in facts_i:
                    """
                    IMPORTANT : embedding entries must be in [-1, +1]
                    thus no matter how it changes over time 
                    (no matter whether it even has a time-varying entry)
                    set entries to -1 and +1 according to parameters 
                    can get its upper bound
                    """
                    output = self.transform_param_intensity[m_i].get_upperbound()  # 1-dim
                    to_aggr.append(output)
                aggred = self.aggregation[b_i](torch.stack(to_aggr, dim=0))
                intensity += aggred
        if hascell:  # if it is an active Transformer
            intensity += active['intensity_cells'][event_type].get_upperbound()
        self.cache_inten_bound[event_type] = \
            self.activation['softplus'](intensity)  # size : 1

    def compute_new_cell(self, c, e, cdb, active, time, for_intensity_dim=False):
        # To re-use the attention code, we will firstly insert a cell with "state" = None
        # Do not worry about that since we will only use the most recent embeddings
        # we will update this state after we complete the state computation
        e = self.datalog.aug_term(e)
        if e not in self.cache_emb:
            self.compute_embedding_(e, cdb, active)
        if for_intensity_dim:
            cell = active['intensity_cells'][c]
            key_fn = self.see_param_intensity_key
            value_fn = self.see_param_intensity_value
        else:
            cell = active['cells'][c]
            key_fn = self.see_param_key
            value_fn = self.see_param_value

        old_history_len = cell.get_history_len()
        for r_i, b_facts_i in cdb.cells[c].see_edges[e].items():
            for b_i, facts_i in b_facts_i.items():
                for body_i, m_i in facts_i:
                    to_cat = [self.cache_emb[e]]
                    # please note that, when creating new cells, in order to re-use compute_embedding_(),
                    # we first input all information except "state" embeddings
                    # the "state" embeddings will be computed at the end of this function
                    template = {
                        "time": time,
                        # only retrieve the most recent event embedding
                        "Event": self.filter_emb(self.cache_emb[e]),
                        "mparam_i": m_i,
                        "betaparam_i": b_i,
                        "rule_i": r_i
                    }
                    for subgoal in body_i:
                        if subgoal not in self.cache_emb:
                            self.compute_embedding_(subgoal, cdb, active)
                        to_cat.append(self.cache_emb[subgoal])
                    cated = torch.cat(to_cat, dim=-1)
                    # -1 for most recent time
                    cated = cated[:, -1, :]

                    # multi-layer K & V are all evaluated at the same time
                    # +1 for 0-th layer
                    _cated_concat_time = self._concat_time_for_key_and_value(cated, [time, ] * (self.layer + 1))
                    # now it is exact key & val
                    # we solve layer-3 for components, but for key & val, we only use up to layer-2
                    # layer-3 is for :- rules during sum_r
                    template["Key"] = []
                    template["Value"] = []
                    for i in range(self.layer):
                        template["Key"].append(key_fn[m_i][i](_cated_concat_time[i]))
                        template["Value"].append(value_fn[m_i][i](_cated_concat_time[i]))
                    template["Key"] = torch.stack(template["Key"], dim=0)
                    template["Value"] = torch.stack(template["Value"], dim=0)
                    cell.update(time, template, True)

    def filter_emb(self, x):
        """
        embedding size : # times * 6D
        """
        d = x.dim()
        if d == 1:
            return x
        elif d == 2:
            # in our setting, most recent time is the actual event time
            return x[-1, :]
        elif d == 3:
            # we have 3-dim : layer * times * 6D
            return x[-1, :, :]
        else:
            raise Exception(f"what embedding is? {x}")
