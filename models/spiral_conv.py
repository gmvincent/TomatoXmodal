"""
Spiral Convolution layer adapted from:

    Gong, S., et al. "SpiralNet++: A Fast and Highly Efficient Mesh Convolution Operator."
    Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops, 2019.

Original implementation (MIT License):
    https://github.com/sw-gong/spiralnet_plus/blob/master/conv/spiralconv.py
"""

import torch
import torch.nn as nn


class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1):
        super(SpiralConv, self).__init__()
        self.dim = dim
        #self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        #self.seq_length = indices.size(1)

        #self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        
        self.seq_legnth = None
        self.layer = None
        

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x, indices):
        n_nodes, seq_len = indices.size()
        bs = x.size(0) if x.dim() == 3 else None
        self.seq_length = seq_len
        self.indices = indices
        
        if self.layer is None:
            self.layer = nn.Linear(self.in_channels * seq_len, self.out_channels).to(x.device)
            self.reset_parameters()
            
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))
        x = self.layer(x)
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)