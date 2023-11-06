from torch import nn
import torch
import numpy as np
from anhp.neural.aggr import Aggregation

# [Chenghao] : why we do not inherit from anhp.neural.aggr.Aggregation?
# well, that is because AttentionAggregation is doing quite different things
# and we do not want to have a complex inheritance link
class ScaledDotAttention(nn.Module):
    def __init__(self, ):
        super(ScaledDotAttention, self).__init__()
        # self.name = name
        self.eps = np.finfo(float).eps
        self.max = torch.finfo().max
        # device = device or 'cpu'
        # self.device = torch.device(device)
        self.act_fn = nn.Softmax(dim=-1)
        # self.aggr_fn = Aggregation(name + "_aggr", device)
    #
    # def use_sum(self, input):
    #     return torch.sum(input, dim=0)
    #
    # def use_extreme(self, input):
    #     ma, _ = torch.max(input, dim=0)
    #     mi, _ = torch.min(input, dim=0)
    #     ma_idx = (torch.abs(ma) >= torch.abs(mi))
    #     y = mi
    #     y[ma_idx] = ma[ma_idx]
    #     return y
    #
    def forward(self, Query, Key, Value, mask=None):
        # [Chenghao]: according to the code of aggr, I assume the input size should be
        # Key and Value: (#{to_be_aggregated}, dim)
        # Query: (#{query}, dim)
        hidden_size = Key.size()[1]
        # return self.aggr_fn(
        if mask is None:
            return \
                torch.mm(self.act_fn(
                    torch.mm(Query, torch.transpose(Key, 0, 1) ) / np.sqrt(hidden_size)
                ), Value)
        else:
            assert mask.dim() == 2 and mask.size(0) == Query.size(0) \
                   and mask.size(1) == Key.size(1), \
                f"masking size mismatch in attention layer:" \
                f" mask_dim: {mask.dim()} " \
                f" mask_size: {mask.size()} " \
                f" query_size: {Query.size()} " \
                f" Key_size: {Key.size()}"
            energy = self.act_fn(
                torch.matmul(Query, Key.T) / np.sqrt(hidden_size)
            ) * mask
            att = energy / (torch.sum(energy, dim=-1, keepdim=True))
            return att.matmul(Value)

        # )




