import math
import functools

import torch
import torch.nn as nn
import torch.distributed as dist
from accessory.util import misc


import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
    copy_to_model_parallel_region,
    reduce_from_model_parallel_region
)
from loguru import logger

def init_env():
    # define the model
    misc.init_distributed_mode()
    fs_init.initialize_model_parallel(dist.get_world_size())

default_linear_init = functools.partial(nn.init.kaiming_uniform_, a=math.sqrt(5))
class ParallelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ParallelLinear, self).__init__()
        # Divide the weight matrix along the last dimension.
        self.col = ColumnParallelLinear(
            in_features, out_features, bias=bias, gather_output=False, init_method=default_linear_init)
        self.row = RowParallelLinear(out_features, in_features, bias=bias, input_is_parallel=True, init_method=default_linear_init)

    def forward(self, x):
        rank = torch.distributed.get_rank()
        print(f"rank{rank}-x.shape={x.shape}")
        out = self.col(x)
        print(f"rank{rank}-self.col.weight.shape={self.col.weight.shape}")
        print(f"rank{rank}-self.row.weight.shape={self.row.weight.shape}")
        print(f"rank{rank}-out1.shape={out.shape}")
        out = self.row(out)
        print(f"rank{rank}-out2.shape={out.shape}")
        return out

def main():
    init_env()
    rank = torch.distributed.get_rank()
    print(f'Start1 from {rank}')
    x = torch.rand(2, 10, 768).to(rank)
    net = ParallelLinear(768, 768).to(rank)
    y = net(x)

if __name__ == '__main__':
    main()
