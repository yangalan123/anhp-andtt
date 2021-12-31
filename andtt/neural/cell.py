import torch
import numpy as np

class TransformerCell(object):
    """
    NOTE : not inherent nn.Module can make creation and copy faster
    """
    def __init__(self, name, cell_dim, cell_zero, time_created, device=None, memory_size=50):
        super(TransformerCell, self).__init__()
        device = device or 'cpu'
        self.device = torch.device(device)
        self.name = name
        self.cell_dim = cell_dim
        assert cell_dim >= 1, "cell dim must > 0"
        self.cell_zero = cell_zero
        self.cell_mask = torch.ones(cell_dim, device=device)
        for c in self.cell_zero:
            self.cell_mask[c] = 0.0
        self.time_created = time_created
        self.max = torch.finfo().max
        self.min = torch.finfo().min
        self.eps = 1e-7
        assert memory_size > 0 and type(memory_size) == int, f"memory size of the cell {name} is not a natural number"
        self.memory_size = memory_size
        self.cache_mparam_dict = dict()
        self.cache_betaparam_dict = dict()
        self.detach_counter = 0
        self.reset()

    def reset(self):
        """
        time of last time the states are updated
        In Transformer-NDTT, there is no "decay"
        K, Q, V only need the current state

        We call each term in history as a "record",
        in each record, the "state" is the "updated" states -- lim_{t->s^+}[[h]](s),
        where t is the "time" of each record
        """
        self.time_updated = self.time_created
        self.history = [
            {
                'state': torch.zeros(size=[self.cell_dim], dtype=torch.float32, device=self.device),
                'time': -1,
                # any number significantly smaller than created time is okay -- it is only a warning signal
                # never use this "time" -- it is just intended for representing the left limit

                # At creating the neural 'cell' (instead of the logic cell), there is no any real K, Q, V and Event
                # If the fact is created by the update rule, then we will use "initial state" as the
                # "most recently updated state" of fact --> [[h]] (s^+), which is used in create "Query"
            }
        ]
        self.cache_mparam_dict = dict()
        self.cache_betaparam_dict = dict()

    def detach(self):
        """
        detach the history graph
        """
        for i in range(self.detach_counter, len(self.history)):
            for key in self.history[i]:
                if isinstance(self.history[i][key], torch.Tensor):
                    self.history[i][key] = self.history[i][key].detach()
        self.detach_counter = len(self.history)

    def update_cache_param_dict(self, cell, idx):
        betaparam_i, mparam_i = cell["betaparam_i"], cell["mparam_i"]
        if betaparam_i not in self.cache_betaparam_dict:
            self.cache_betaparam_dict[betaparam_i] = set()
        if mparam_i not in self.cache_mparam_dict:
            self.cache_mparam_dict[mparam_i] = set()
        self.cache_mparam_dict[mparam_i].add(idx)
        self.cache_betaparam_dict[betaparam_i].add(idx)


    def update(self, time, cell, append=False, idx=-1):
        """
        this function updates the configuration of this partition
        append = False to save memory
        """
        # we want to allow "partial-update" to re-use the code ("compute_embedding_") for doing attention
        # that is, during cell updating time in program running, there will be several records in the history
        # that do not have the "state" -- that does not matter, since the computation of state [[h]](t) at time t
        # will and definitely should not involve [[h]](t)
        # our attention codes will make sure we will never use that time
        # please also look at search_most_recent_until_t(), we intentionally doing "-1" when returning the indexes

        if "state" in cell:
            assert cell['state'].size(0) == self.cell_dim, \
                f"Dimension mismatch : {cell['state'].size(0)} vs. {self.cell_dim}"
        self.time_updated = time
        """
        new cells are aggregated values so might be too large or too small
        """
        new_cell = {}
        for k, v in cell.items():
            # Note that now we have to store much more information than before -- also contains non-Tensor info
            if isinstance(v, torch.Tensor):
                new_cell[k] = torch.clamp(v, min=self.min, max=self.max)
            else:
                new_cell[k] = v

        if append:
            self.history.append(new_cell)
        else:
            for key in new_cell:
                if key in self.history[idx] and isinstance(new_cell[key], torch.Tensor):
                    assert False, "wtf_in_cell"
                self.history[idx][key] = new_cell[key]
        _idx = idx
        if _idx == -1:
            _idx += len(self.history)
        if "betaparam_i" in cell and "mparam_i" in cell:
            self.update_cache_param_dict(new_cell, _idx)

    def retrospect(self, idx=-1):
        """
        this function pops most recent memory
        """
        return self.history[idx]

    def get_history_len(self):
        return len(self.history)

    def get_upperbound(self):
        """
        return a value that can't be exceeded no matter what time it is
        i.e., for each dim, get max(start, target)
        """
        assert len(self.history) > 1, 'want to compute the upperbound before the event occurs?'
        buf = []
        for i in range(1, len(self.history)):
            # new bound: for each layer, compute max(value_j^{(l)})
            buf.append(self.history[i]["Value"][-1])
        return torch.stack(buf, dim=0).max(1)[0].detach()


    def search_most_recent_until_t(self, times, side="left"):
        # now the input `times` is sorted, which allows more efficient computation
        _times = times
        if isinstance(times, torch.Tensor):
            if _times.is_cuda:
                _times = times.cpu().numpy()
            else:
                _times = times.numpy()
        #  Now only search among most recent memory_size history
        #  why not bind it to detach_cells()? since detach and shrink should be totally two different things!
        #  you never want to have the risk that, before you backprop, some related Tensor has been "removed"
        #  as a matter of fact, shrinking only wants to cut down the computational burden, not want to interfere too much with comp graph
        if len(self.history) > self.memory_size + 1:
            truncated_history = self.history[:1] + self.history[1:][-self.memory_size:]
        else:
            truncated_history = self.history
        history_time = [x["time"] for x in truncated_history]
        # 1) when finding [[h]] ( s^+(t) ), must be left side: a[i-1] < x <= a[i]
        # when computing query for time t, you should never use the state embedding after t!!!
        # you can only use the state embedding that is most recently to t but before t (t^-)
        # 2) when finding e(s), must be right side: a[i-1] <= x < a[i]

        _ret = np.searchsorted(history_time, _times, side=side)

        if side == "left":
            # approximately the final updated time
            _ret[(abs(_times - self.time_updated) / (self.time_updated + self.eps) < self.eps)
                 & ((_ret - 1) > 0)] = len(truncated_history) - 1

        else:
            # approximately the final updated time
            _ret[abs(_times - self.time_updated) / (self.time_updated + self.eps) < self.eps] = len(truncated_history)


        # the following condition must hold -- even if we sampled the created time,
        # in computing embeddings, the returned results of this func will only be used to find [[h]](s^+)
        # so for the created time, since we have set history[0]["time"] < created_time,
        # the search algorithm will turn to the idx corresponding to zero embedding, that is what we want
        # why minus 1? please look at 1) and 2)
        assert ((_ret - 1) >= 0).all()
        _ret = _ret - 1

        # do merging
        x = _ret[0]
        res = []
        _res = []
        for _history_id, time in zip(_ret, times):
            if _history_id == x:
                _res.append(time)
            else:
                res.append({"history_id": x, "times": torch.Tensor(_res)})
                x = _history_id
                _res = [time, ]
        res.append({"history_id": x, "times": torch.Tensor(_res)})
        return res, truncated_history


    def get_cached_event_embeddings_dict(self):
        ret = dict()
        for i, item in enumerate(self.history):
            if i == 0:
                # remember that history[0] has no event!
                continue
            else:
                ret[i] = item["Event"]
        return ret


    def cuda(self, device=None):
        device = device or 'cuda:0'
        self.device = torch.device(device)
        assert self.device.type == 'cuda'
        super().cuda(self.device)

    def cpu(self):
        self.device = torch.device('cpu')
        super().cuda(self.device)

    def copy(self):
        """
        between shallow and deep copy : only copy initial values
        """
        cp_obj = TransformerCell(
            self.name, self.cell_dim, self.cell_zero, self.time_created, self.device)
        return cp_obj

    def deepcopy(self):
        """
        deep copy : copy current (i.e. most recent) values
        """
        cp_obj = self.copy()
        cp_obj.time_updated = self.time_updated
        for k, v in self.history[-1].items():
            """
            .detach but not .clone 
            cuz we do not want the graph info behind it
            why no graph? cp_obj might be passed into other processes
            """
            if isinstance(cp_obj.history[-1][k], torch.Tensor):
                cp_obj.history[-1][k] = v.detach()
        return cp_obj

    def __repr__(self):
        s0 = 'TransformerCell:\n'
        s0 += '{} : cell_dim={}, cell_zero={}, time_created={}'.format(
            self.name, self.cell_dim, self.cell_zero, self.time_created
        )
        s1 = f'\ntime_updated={self.time_updated}, history={self.history}'
        return s0 + s1



