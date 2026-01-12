import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

class DenseGCNLayer(nn.Module):
    """
    A single GCN layer that takes concatenated input features and produces new features.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = BatchNorm(out_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        return F.relu(x)


class DCBlock(nn.Module):
    """
    Dense-Connected GCN Block (like DenseNet dense block but using GCNConv).
    Each layer concatenates all previous outputs (including block input).
    """
    def __init__(self, in_channels, growth_channels, n_layers):
        """
        in_channels: channels entering block
        growth_channels: output channels added by each layer (k)
        n_layers: number of dense layers in block
        """
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # input channels grow as we add layers
            layer_in = in_channels + i * growth_channels
            self.layers.append(DenseGCNLayer(layer_in, growth_channels))

    def forward(self, x, edge_index):
        features = [x]
        for layer in self.layers:
            x_cat = torch.cat(features, dim=-1)
            out = layer(x_cat, edge_index)
            features.append(out)
        # output concatenates all features (optionally compress)
        return torch.cat(features, dim=-1)


class TransitionLayer(nn.Module):
    """Optional compressing layer between DC blocks"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: (num_nodes, channels)
        x = self.linear(x)
        x = self.bn(x)
        return F.relu(x)


class MDC_GCN(nn.Module):
    def __init__(self, in_channels, num_classes,
                 dc_config=[(32,3), (64,3), (128,3), (64,3)],  # list of (growth_channels, n_layers) per DC block
                 transition_channels=None,    # list of compress channels after each DC block
                 final_gcn_channels=128,
                 dropout=0.3):
        """
        dc_config: e.g., [(k1, L1), (k2, L2)] where each block uses growth k and L layers.
        transition_channels: optional list with length = len(dc_config) to compress after each block
        """
        super().__init__()
        self.in_proj = nn.Linear(in_channels, dc_config[0][0])  # project inputs to a base dim

        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList() if transition_channels is not None else None

        cur_channels = dc_config[0][0]
        for idx, (growth, n_layers) in enumerate(dc_config):
            # For first block, in_channels is cur_channels; subsequent blocks get compressed channels as input
            block = DCBlock(cur_channels, growth, n_layers)
            self.blocks.append(block)
            cur_channels = cur_channels + growth * n_layers  # channel count after concatenation

            if transition_channels is not None:
                out_ch = transition_channels[idx]
                self.transitions.append(TransitionLayer(cur_channels, out_ch))
                cur_channels = out_ch  # compressed for next block

        # final GCN to fuse per-node features (choose out size)
        self.final_gcn = GCNConv(cur_channels, final_gcn_channels)
        self.final_bn = BatchNorm(final_gcn_channels)

        self.classifier = nn.Sequential(
            nn.Linear(final_gcn_channels, final_gcn_channels//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_gcn_channels//2, num_classes)
        )

    def forward(self, x, edge_index, batch, return_features=False):
        """
        data: torch_geometric.data.Data or Batch with:
          x: node features (N_nodes_total, in_channels)
          edge_index: adjacency (2, num_edges)
          batch: batch vector if batched graphs
        """
        #x, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)

        x = self.in_proj(x)  # project

        for i, block in enumerate(self.blocks):
            x = block(x, edge_index)
            if self.transitions is not None:
                x = self.transitions[i](x)

        x = self.final_gcn(x, edge_index)
        x = self.final_bn(x)
        x = F.relu(x)

        # global mean pooling per graph
        if batch is None:
            # single graph: create batch of zeros
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        g = global_mean_pool(x, batch)  # (batch_size, final_gcn_channels)

        out = self.classifier(g)        # (batch_size, num_classes)
        if return_features:
            return g, out
        else:
            return out

