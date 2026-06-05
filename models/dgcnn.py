"""
DGCNN Model adapted from:

    Wang, Y., Sun, Y., Liu, Z., Sarma, S. E., Bronstein, M. M., & Solomon, J. M. (2019).
    Dynamic graph cnn for learning on point clouds. ACM Transactions on Graphics (tog), 
    38(5), 1-12.

Original implementation (MIT License):
    https://github.com/WangYueFt/dgcnn
    
Reduced model capacity, removed `self.conv5` and related layers
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F


def knn(x, k):
    # x: [B, C, N]
    x_t = x.transpose(2, 1).contiguous()  # [B, N, C]

    inner = -2 * torch.matmul(x_t, x_t.transpose(2, 1))  # [B, N, N]
    xx = torch.sum(x_t ** 2, dim=-1, keepdim=True)

    dist = -xx - inner - xx.transpose(2, 1)

    idx = dist.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None, xyz=None):
    B, C, N = x.shape
    x = x.view(B, -1, N)
    
    if idx is None:
        # Build graph from xyz if provided, else from x itself
        graph_x = xyz if xyz is not None else x
        idx = knn(graph_x, k=k)

    x_t = x.transpose(2, 1).contiguous()  # [B, N, C]

    if idx is None:
        idx = knn(x, k=k)

    device = x.device

    idx_base = torch.arange(B, device=device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)

    x_flat = x_t.view(B * N, C)
    feature = x_flat[idx, :]
    feature = feature.view(B, N, k, C)

    x_center = x_t.view(B, N, 1, C).expand(-1, -1, k, -1)

    feature = torch.cat([feature - x_center, x_center], dim=-1)

    return feature.permute(0, 3, 1, 2).contiguous()

class DGCNN(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int, k: int=10, include_spectral: bool=True):
        super(DGCNN, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.include_spectral = include_spectral
        
        self.k = k # Num of nearest neighbors to use
        
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.bn4 = torch.nn.BatchNorm1d(512)

        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   torch.nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   torch.nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   torch.nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = torch.nn.Sequential(torch.nn.Conv1d(256, 512, kernel_size=1, bias=False),
                                   self.bn4,
                                   torch.nn.LeakyReLU(negative_slope=0.2))
        
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.bn6 = torch.nn.BatchNorm1d(128)
        
        self.linear1 = torch.nn.Linear(512*2, 256, bias=False)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, num_classes)

        self.dp1 = torch.nn.Dropout(p=0.5)
        self.dp2 = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        batch_size = x.size(0)
        
        if self.include_spectral and x.shape[1] > 3:
            xyz = x[:, :3, :]      # geometry for graph structure
            feats = x[:, 3:-3, :]    # spectral features
        else:
            xyz = x[:, :3, :]
            feats = None
        
        
        x = get_graph_feature(feats, k=self.k, xyz=xyz)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (B, 128, N)

        x = torch.cat((x1, x2, x3), dim=1)    # (B, 256, N) — no x4

        x = self.conv4(x)                      # (B, 512, N)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # (B, 512)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # (B, 512)
        x = torch.cat((x1, x2), dim=1)         # (B, 1024)

        x = F.leaky_relu(self.bn5(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn6(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x