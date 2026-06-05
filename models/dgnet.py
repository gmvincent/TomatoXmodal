"""
DGNet Model adapted from:

    Li, X. L., et al. (2023). Mesh neural networks based on dual graph pyramids.
    IEEE Transactions on Visualization and Computer Graphics, 30(7), 4211-4224.

Original implementation (MIT License):
    https://github.com/li-xl/DGNet

PyTorch port: Jittor-specific ops (jt.Function, jt.code CUDA kernels, feats.reindex)
have been replaced with pure PyTorch equivalents. Pool/Unpool custom CUDA kernels are
reimplemented using scatter/gather. with_spatial=True path is disabled (requires
Jittor-specific spatial indexing not used in the basic pipeline).
"""
import torch
import numpy as np 
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MeshTensor:
    """
    Holds per-face features and all mesh topology needed by DGNet layers.

    feats : [N, C, F]  — batched per-face feature tensor
    face_adjacency : [N, F, 3]  — for each face, indices of its 3 neighbours (-1 = boundary)
    Fs : [N]  — number of valid faces per batch item
    level : int  — current pyramid level (0 = finest)

    Pool/unpool bookkeeping (populated by MeshPool, consumed by MeshUnpool):
        pool_mask, next_Mf, last_pool_mask, last_face_adjacency, last_Mf, last_indexes
    """
    feats:              torch.Tensor
    face_adjacency:     Optional[torch.Tensor] = None
    Fs:                 Optional[torch.Tensor] = None
    level:              int = 0

    # pool bookkeeping
    pool_mask:          Optional[torch.Tensor] = None
    next_Mf:            Optional[int]          = None
    last_pool_mask:     Optional[torch.Tensor] = None
    last_face_adjacency: Optional[torch.Tensor] = None
    last_Mf:            Optional[int]          = None
    last_indexes:       Optional[torch.Tensor] = None

    def updated(self, **kwargs) -> "MeshTensor":
        """Return a shallow copy with selected fields replaced."""
        import dataclasses
        return dataclasses.replace(self, **kwargs)
    
    
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def reindex_feats(feats: torch.Tensor, index: torch.Tensor, max_f: int) -> torch.Tensor:
    """
    Gather features at positions given by `index`.

    feats : [N, C, F]
    index : [N, max_f]   (values in [0, F-1], -1 = invalid / padding)
    returns [N, C, max_f]
    """
    N, C, F = feats.shape
    
    # mask for invalid indices
    valid_mask = index >= 0                          # [N, max_f]
    safe_index = index.clone()
    safe_index[~valid_mask] = 0
    
    # expand for gather
    gather_index = safe_index.unsqueeze(1).expand(-1, C, -1)   # [N, C, max_f]
    out = torch.gather(feats, dim=2, index=gather_index)
    
    # zero out invalid positions
    out = out * valid_mask.unsqueeze(1).float()
    return out

def _gather_adj_feats(feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """
    Collect features of neighbouring faces.

    feats : [N, C, F]
    adj   : [N, F, K]   neighbour face indices (-1 = no neighbour)
    returns [N, C, F, K]
    """
    N, C, F = feats.shape
    K = adj.shape[2]
    
    # mask for invalid indices
    valid = adj >= 0                                  # [N, F, K]
    safe_adj = adj.clone()
    safe_adj[~valid] = 0

    # gather over dim=2 (F dim)
    gathered = feats[:, :, safe_adj.view(N, -1)].view(N, C, F, K)  # simpler
    
    # zero out invalid
    gathered = gathered * valid.unsqueeze(1).float()
    return gathered

def dilated_face_adjacencies(FAF: torch.Tensor, dilation: int) -> torch.Tensor:
    """
    Pure-Python / PyTorch reimplementation of the Jittor CUDA kernel.
    Walks `dilation` hops along the dual graph starting from each face-edge.

    FAF : [N, F, 3]  face-adjacency (standard ring-1)
    returns [N, F, 3]
    """
    if dilation <= 1:
        return FAF

    N, F, _ = FAF.shape
    device = FAF.device
    DFA = FAF.clone()

    # We need to walk the adjacency graph.
    # For each (n, f, k): start at face f, go to neighbour k, then walk dilation-1 more steps.
    # This is done on CPU (index walking is not efficiently parallelisable in pure PyTorch).
    faf_np = FAF.cpu().numpy()
    dfa_np = DFA.cpu().numpy().copy()

    for bs in range(N):
        for f in range(F):
            for k in range(3):
                a = f
                b = int(faf_np[bs, f, k])
                for d in range(1, dilation):
                    if b < 0:
                        break
                    # find which neighbour of b points back to a
                    neighs = faf_np[bs, b]
                    if neighs[0] == a:
                        i = 0
                    elif neighs[1] == a:
                        i = 1
                    else:
                        i = 2
                    a = b
                    if (d & 1) == 0:   # even: go to next
                        b = int(neighs[i + 1] if i < 2 else neighs[0])
                    else:              # odd: go to previous
                        b = int(neighs[i - 1] if i > 0 else neighs[2])
                dfa_np[bs, f, k] = b

    return torch.tensor(dfa_np, dtype=FAF.dtype, device=device)


def pool_func(feats: torch.Tensor,
                    pool_mask: torch.Tensor,
                    adj: torch.Tensor) -> torch.Tensor:
    """
    Max-pool each face with its unmasked neighbours.

    feats : [N, C, F]
    pool_mask : [N, F]   1 = kept face, 0 = dropped
    adj : [N, F, 3]
    returns : [N, C, F]
    """
    N, C, F = feats.shape
    out = feats.clone()

    for n in range(N):
        for f in range(F):
            if pool_mask[n, f] == 1:
                for k in range(3):
                    a = int(adj[n, f, k].item())
                    if a >= 0 and pool_mask[n, a] == 0:
                        better = feats[n, :, a] > out[n, :, f]
                        out[n, :, f] = torch.where(better, feats[n, :, a], out[n, :, f])
    return out


def unpool_func(feats: torch.Tensor,
                      pool_mask: torch.Tensor,
                      adj: torch.Tensor,
                      bilinear: bool = True) -> torch.Tensor:
    """
    Unpool: fill dropped faces by averaging their surviving neighbours' features.

    feats : [N, C, F]
    pool_mask : [N, F]
    adj : [N, F, 3]
    returns : [N, C, F]
    """
    N, C, F = feats.shape
    out = feats.clone()

    for n in range(N):
        for f in range(F):
            if pool_mask[n, f] == 0:
                val = torch.zeros(C, device=feats.device, dtype=feats.dtype)
                count = 0
                for k in range(3):
                    a = int(adj[n, f, k].item())
                    if a >= 0 and pool_mask[n, a] == 1:
                        val += feats[n, :, a]
                        count += 1
                        if not bilinear:
                            break
                out[n, :, f] = val / count if count > 0 else torch.zeros(C, device=feats.device)
    return out


# ---------------------------------------------------------------------------
# Basic mesh layers
# ---------------------------------------------------------------------------


class MeshReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, mesh: MeshTensor) -> MeshTensor:
        return mesh.updated(feats=self.relu(mesh.feats))

class MeshDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.dropout = torch.nn.Dropout(p)

    def forward(self, mesh: MeshTensor) -> MeshTensor:
        return mesh.updated(feats=self.dropout(mesh.feats))

class MeshLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, mesh: MeshTensor) -> MeshTensor:
        return mesh.updated(feats=self.conv1d(mesh.feats))
    
class MeshBatchNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(num_features, eps, momentum)

    def forward(self, mesh: MeshTensor) -> MeshTensor:
        return mesh.updated(feats=self.bn(mesh.feats))
    
class MeshInstanceNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.ln = torch.nn.InstanceNorm1d(num_features, eps)

    def forward(self, mesh: MeshTensor) -> MeshTensor:
        return mesh.updated(feats=self.ln(mesh.feats))

def mesh_concat(meshes: list) -> MeshTensor:
    new_feats = torch.cat([m.feats for m in meshes], dim=1)
    return meshes[0].updated(feats=new_feats)


# ---------------------------------------------------------------------------
# Pool / Unpool
# ---------------------------------------------------------------------------


class MeshPool(torch.nn.Module):
    def __init__(self, op: str = "max"):
        super().__init__()
        assert op in ["max", "none"]
        self.op = op
    
    def forward(self, mesh: MeshTensor) -> MeshTensor:
        feats = mesh.feats
        N, C, F = feats.shape

        pool_mask = mesh.pool_mask     # [N, F]
        adj = mesh.face_adjacency      # [N, F, 3]
        max_f = mesh.next_Mf

        if self.op == "max":
            feats = pool_func(feats, pool_mask, adj)

        # Build index mapping old → new (compact) positions
        indexes = -torch.ones((N, max_f), dtype=torch.long, device=feats.device)
        for i in range(N):
            kept = torch.where(pool_mask[i])[0]          # indices of kept faces
            n_kept = kept.shape[0]
            indexes[i, :n_kept] = kept

        feats = reindex_feats(feats, indexes, max_f)

        return mesh.updated(feats=feats, level=mesh.level + 1)
    


class MeshUnpool(torch.nn.Module):
    def __init__(self, mode: str = "nearest"):
        super().__init__()
        self.mode = mode
        assert mode in ["nearest", "bilinear"]

    def forward(self, mesh: MeshTensor) -> MeshTensor:
        feats = mesh.feats                  # [N, C, F_pooled]
        N, C, _ = feats.shape

        pool_mask = mesh.last_pool_mask
        adj = mesh.last_face_adjacency
        max_f = mesh.last_Mf
        indexes = mesh.last_indexes         # [N, F_pooled] → original indices

        feats = reindex_feats(feats, indexes, max_f)
        feats = unpool_func(feats, pool_mask, adj, bilinear=(self.mode == "bilinear"))

        return mesh.updated(feats=feats, level=mesh.level - 1)
          

# ---------------------------------------------------------------------------
# SpatialConv  (with_spatial path disabled — requires Jittor spatial indexing)
# ---------------------------------------------------------------------------


class SpatialConv(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size:  int = 3,
                 dilation: int = 1,
                 stride: int = 1,
                 merge_op: str = "max",
                 groups: int = 1,
                 with_spatial: bool = False,   # spatial path not supported without Jittor
                 radius: float = 0.1,
                 max_sample: int = 20,
                 temp_sample: int = 1000,
                 bias: bool = False):
        super().__init__()
        assert stride == 1 and kernel_size in [1, 3]
        assert merge_op in ["max", "mean"]

        self.kernel_size  = kernel_size
        self.dilation     = dilation
        self.merge_op     = merge_op
        self.with_spatial = False          # force-disable; spatial needs Jittor

        self.conv_f = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                bias=bias, groups=groups)

        if kernel_size > 1:
            self.conv_mf = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                     bias=False, groups=groups)

        # Squeeze-and-excitation channel attention
        se_dim = max(1, out_channels // 16)
        self.se = torch.nn.Sequential(
            torch.nn.Linear(out_channels, se_dim, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(se_dim, out_channels, bias=False),
            torch.nn.Sigmoid(),
        )

    def forward(self, mesh: MeshTensor) -> MeshTensor:
        feats = mesh.feats           # [N, C, F]
        N, C, F = feats.shape

        f1 = self.conv_f(feats)      # [N, out_C, F]

        if self.kernel_size == 3:
            f2 = self.conv_mf(feats)                             # [N, out_C, F]
            adj = mesh.face_adjacency                            # [N, F, 3]
            if self.dilation > 1:
                adj = dilated_face_adjacencies(adj, self.dilation)

            # gather neighbour features: [N, out_C, F, 3]
            N2, C2, F2 = f2.shape
            K = adj.shape[2]
            valid = adj >= 0                                     # [N, F, K]
            
            safe_adj = adj.clone()
            safe_adj[safe_adj < 0] = 0                          # [N, F, K]
            
            # correct gather: for each position (n,c,f,k) take f2[n,c,adj[n,f,k]]
            idx = safe_adj.unsqueeze(1).expand(N2, C2, F2, K)   # [N, C2, F, K]
            f2_exp = f2.unsqueeze(-1).expand(N2, C2, F2, K)     # [N, C2, F, K] — not used directly
            # gather neighbours: for each (n,c,f,k) → f2[n, c, adj[n,f,k]]
            f2_adj = torch.gather(
                f2.unsqueeze(3).expand(N2, C2, F2, K),          # [N, C2, F, K]
                dim=2,
                index=idx,
            )                                                    # [N, C2, F, K]
            f2_adj = f2_adj * valid.unsqueeze(1).float()

            if self.merge_op == "max":
                f2_merged, _ = f2_adj.max(dim=-1)
            else:
                valid_count = valid.sum(dim=-1, keepdim=True).unsqueeze(1).float().clamp(min=1)
                f2_merged = f2_adj.sum(dim=-1) / valid_count.squeeze(-1)
        else:
            f2_merged = 0.0

        feats_out = f1 + f2_merged if isinstance(f2_merged, torch.Tensor) else f1

        # Channel attention via SE
        attn = self.se(feats_out.mean(dim=-1))  # [N, out_C]
        feats_out = attn.unsqueeze(-1) * feats_out

        return mesh.updated(feats=feats_out)


# ---------------------------------------------------------------------------
# Higher-level blocks
# ---------------------------------------------------------------------------


class MLP(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 with_bn: bool = True,
                 dropout: float = 0.,
                 bias:  bool = False):
        super().__init__()
        
        self.mlp = SpatialConv(in_channels, out_channels, kernel_size=kernel_size, bias=bias)
        self.relu = MeshReLU()
        self.bn = MeshBatchNorm(out_channels) if with_bn else torch.nn.Identity()
        self.dropout = MeshDropout(dropout)

    def forward(self, mesh: MeshTensor) -> MeshTensor:
        mesh = self.mlp(mesh)
        mesh = self.relu(mesh)
        mesh = self.bn(mesh) if isinstance(self.bn, MeshBatchNorm) else mesh
        mesh = self.dropout(mesh)
        return mesh

class _MeshSequential(torch.nn.Module):
    """nn.Sequential wrapper that passes MeshTensor through each layer."""
    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            self.add_module(str(i), m)

    def forward(self, mesh: MeshTensor) -> MeshTensor:
        for m in self.children():
            mesh = m(mesh)
        return mesh

class SpatialBlock(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dilation: int = 1,
                 radius: float = None,
                 max_sample: int = 20,
                 temp_sample: int = 1000,
                 merge_op: str = "max"):
        super().__init__()
        with_spatial = radius is not None
        
        self.conv1 = _MeshSequential(
            SpatialConv(in_channels, out_channels,
                        dilation=dilation,
                        with_spatial=with_spatial,
                        radius=radius if radius else 0.1,
                        max_sample=max_sample,
                        temp_sample=temp_sample,
                        merge_op=merge_op),
            MeshReLU(),
            MeshBatchNorm(out_channels),
        )

        self.res1 = _MeshSequential(
            SpatialConv(out_channels, out_channels),
            MeshReLU(),
            MeshInstanceNorm(out_channels),
        )

    def forward(self, mesh: MeshTensor) -> MeshTensor:
        mesh = self.conv1(mesh)
        res  = self.res1(mesh)
        return mesh.updated(feats=mesh.feats + res.feats)


# ---------------------------------------------------------------------------
# DGNet
# ---------------------------------------------------------------------------


class DGNet(torch.nn.Module):
    def __init__(self,
                 in_channels: int = 16,
                 encoder_channels: list = None,
                 dilations: list = None,
                 radius: list = None,
                 dropouts: list = None,
                 cls_dropouts: list = None,
                 decoder_channels: list = None,
                 max_sample: int = 30,
                 temp_sample: int = 1000,
                 merge_op: str = "max",
                 use_pool: bool = False,
                 num_classes: int = 4,
                 include_spectral: bool = False):
        super().__init__()

        if encoder_channels is None:
            encoder_channels = [32, 64, 96, 128, 128]
        if decoder_channels is None:
            decoder_channels = [128, 128, 128, 96, 96]
        if dilations is None:
            dilations = [1, 1, 1, 1]
        if radius is None:
            radius = [0., 0.2, 0.4, 0.8]
        if dropouts is None:
            dropouts = [0., 0., 0., 0.]
        self.include_spectral = include_spectral
        
        self.depth = len(encoder_channels) - 1
        assert len(encoder_channels) == len(decoder_channels)
        assert encoder_channels[-1] == decoder_channels[0]
        assert self.depth == len(radius) == len(dilations) == len(dropouts)

        self.block1 = _MeshSequential(
            MLP(in_channels, encoder_channels[0], with_bn=True),
            SpatialBlock(encoder_channels[0], encoder_channels[0]),
        )

        if use_pool:
            self.pool   = MeshPool(op="max")
            self.unpool = MeshUnpool(mode="bilinear")
        else:
            self.pool   = torch.nn.Identity()
            self.unpool = torch.nn.Identity()

        self.encoders = torch.nn.ModuleList([
            SpatialBlock(encoder_channels[i], encoder_channels[i + 1],
                         dilation=dilations[i],
                         radius=radius[i] if radius[i] > 0 else None,
                         max_sample=max_sample,
                         temp_sample=temp_sample,
                         merge_op=merge_op)
            for i in range(self.depth)
        ])

        self.decoders = torch.nn.ModuleList([
            SpatialBlock(decoder_channels[i] + encoder_channels[self.depth - 1 - i],
                         decoder_channels[i + 1],
                         dilation=dilations[-(i + 1)],
                         radius=radius[-(i + 1)] if radius[-(i + 1)] > 0 else None,
                         max_sample=max_sample,
                         temp_sample=temp_sample,
                         merge_op=merge_op)
            for i in range(self.depth)
        ])

        if cls_dropouts is None:
            self.predict = MeshLinear(decoder_channels[-1], num_classes)
        else:
            layers = []
            for d in cls_dropouts:
                layers.append(MeshLinear(decoder_channels[-1], decoder_channels[-1]))
                layers.append(MeshDropout(d))
            layers.append(MeshLinear(decoder_channels[-1], num_classes))
            self.predict = _MeshSequential(*layers)


    def forward(self, data, return_features=False) -> torch.Tensor:
        vert, spec_feats, faces, centers, normals, ring_1, ring_2, ring_3 = (
            data["verts"],      # list of [V_i, 3] tensors
            data["face_feats"], # [B, 3, F]
            data["faces"],      # [B, F, 3]
            data["centers"],    # [B, 3, F]
            data["normals"],    # [B, 3, F]
            data["ring_1"],     # [B, F, 3]
            data["ring_2"],     # [B, F, 6]
            data["ring_3"],     # [B, F, 12]
        )
        
        # Build per-face features: 
        if self.include_spectral:
            feats = torch.cat([normals, centers, spec_feats], dim=1)   # [N, 10, F]
        else:
            feats = torch.cat([normals, centers], dim=1)        # [N, 6, F]

        # face_adjacency: ring_1 is [N, F, 3], clamp OOB to -1
        N, F, _ = ring_1.shape
        adj = ring_1.clone()
        adj[adj >= F] = -1

        Fs = torch.full((N,), F, dtype=torch.long, device=feats.device)
    
        mesh = MeshTensor(feats=feats, face_adjacency=adj, Fs=Fs)
        mesh = self.block1(mesh)

        enc_meshes = [mesh]

        # Encoder
        for i in range(self.depth):
            if not isinstance(self.pool, torch.nn.Identity):
                mesh = self.pool(mesh)
            mesh = self.encoders[i](mesh)
            enc_meshes.append(mesh)

        if return_features:
            latent = mesh.feats
        
        # Decoder
        for i in range(self.depth):
            if not isinstance(self.unpool, torch.nn.Identity):
                mesh = self.unpool(mesh)
            enc_mesh   = enc_meshes[self.depth - i - 1]
            concat_feats = torch.cat([mesh.feats, enc_mesh.feats], dim=1)
            mesh = self.decoders[i](enc_mesh.updated(feats=concat_feats))

        mesh = self.predict(mesh)
        out = mesh.feats                   # [N, num_classes, F]
        
        if return_features:
            return latent.mean(dim=-1), out.mean(dim=-1)   
        else:
            return out.mean(dim=-1)            # [N, num_classes]