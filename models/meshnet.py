"""
MeshNet++ Model adapted from:

    Singh, V. V., et al. (2021, October). MeshNet++: A Network with 
    a Face. In ACM multimedia (pp. 4883-4891).

Original implementation (MIT License):
    https://github.com/VimsLab/MeshNet2
"""

import torch

from models.meshnet_helpers import PointDescriptor, NormalDescriptor, SpectralDescriptor
from models.meshnet_helpers import ConvSurface, MeshBlock, MaxPoolFaceFeature

class MeshNet2(torch.nn.Module):
    """ MeshNet++ Model"""
    def __init__(self, num_faces, num_cls, pool_rate, num_kernel=64, blocks=[3, 4, 4], num_samples_per_neighbor=4, rs_mode="Weighted", conv_num_kernel=64, include_spectral=False):
        """
        Args:
            num_faces: number of mesh faces
            num_cls: number of classes in dataset
            num_kernel: 
        """
        # Setup
        super(MeshNet2, self).__init__()
        self.pool_rate = pool_rate
        self.include_spectral = include_spectral
        
        self.point_descriptor = PointDescriptor(num_kernel=num_kernel)
        self.normal_descriptor = NormalDescriptor(num_kernel=num_kernel)
        self.spectral_descriptor = SpectralDescriptor(num_kernel=num_kernel)
        self.conv_surface_1 = ConvSurface(num_faces=num_faces, num_neighbor=3, num_samples_per_neighbor=num_samples_per_neighbor, rs_mode=rs_mode, num_kernel=conv_num_kernel)
        self.conv_surface_2 = ConvSurface(num_faces=num_faces, num_neighbor=6, num_samples_per_neighbor=num_samples_per_neighbor, rs_mode=rs_mode, num_kernel=conv_num_kernel)
        self.conv_surface_3 = ConvSurface(num_faces=num_faces, num_neighbor=12, num_samples_per_neighbor=num_samples_per_neighbor, rs_mode=rs_mode, num_kernel=conv_num_kernel)

        
        in_channel = num_kernel * 2 + conv_num_kernel * 3 
        in_channel += num_kernel if include_spectral else 0 # if we include the 4 spectral channels

        self.mesh_block_1 = MeshBlock(in_channel=in_channel,
                                      num_block=blocks[0],
                                      growth_factor=num_kernel,
                                      num_neighbor=3)
        in_channel = in_channel + blocks[0] * num_kernel
        self.max_pool_fea_1 = MaxPoolFaceFeature(in_channel=in_channel, num_neighbor=3)

        self.mesh_block_2 = PsuedoMeshBlock(in_channel=in_channel,
                                            num_block=blocks[1],
                                            growth_factor=num_kernel,
                                            num_neighbor=6)
        in_channel = in_channel + blocks[1] * num_kernel
        self.max_pool_fea_2 = MaxPoolFaceFeature(in_channel=in_channel, num_neighbor=6)

        self.mesh_block_3 = PsuedoMeshBlock(in_channel=in_channel,
                                            num_block=blocks[2],
                                            growth_factor=num_kernel,
                                            num_neighbor=12)
        in_channel = in_channel + blocks[2] * num_kernel

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channel, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256, num_cls)
        )

        print('Spatial descriptor number of learnable kernels: {0}'.format(num_kernel))
        print('Structural descriptor number of learnable kernels: {0}'.format(conv_num_kernel))
        print('Structural descriptor resampling mode: {0}'.format(rs_mode))
        print('MeshNet2 pool rate: {0}'.format(self.pool_rate))

    def forward(self, data):
        """
        Args:
            verts: padded mesh vertices
            [num_meshes, ?, 3]

            faces: faces in mesh/es
            [num_meshes, num_faces, 3]

            centers: face center of mesh/es
            [num_meshes, num_faces, 3]

            normals: face normals of mesh/es
            [num_meshes, num_faces, 3]

            ring_1: 1st Ring neighbors of faces
            [num_meshes, num_faces, 3]

            ring_2: 2nd Ring neighbors of faces
            [num_meshes, num_faces, 6]

            ring_3: 3rd Ring neighbors of faces
            [num_meshes, num_faces, 12]

        Returns:
            cls: predicted class of the input mesh/es
        """
        verts, vert_feats, face_feats, faces, centers, normals, ring_1, ring_2, ring_3 = (
            data["verts"],        # list of [V_i, 3] tensors
            data["vert_feats"],   # [B, V_i, 4]
            data["face_feats"],   # [B, F, 4]
            data["faces"],        # [B, F, 3]
            data["centers"],      # [B, 3, F]
            data["normals"],      # [B, 3, F]
            data["ring_1"],       # [B, F, 3]
            data["ring_2"],       # [B, F, 6]
            data["ring_3"],       # [B, F, 12]
        )
        
        # Face center features
        points_fea = self.point_descriptor(centers=centers)

        # Face normal features
        normals_fea = self.normal_descriptor(normals=normals)

        # Surface features from 1-Ring neighborhood around a face
        surface_fea_1 = self.conv_surface_1(verts=verts,
                                            faces=faces,
                                            ring_n=ring_1,
                                            centers=centers)

        # Surface features from 2-Ring neighborhood around a face
        surface_fea_2 = self.conv_surface_2(verts=verts,
                                            faces=faces,
                                            ring_n=ring_2,
                                            centers=centers)

        # Surface features from 3-Ring neighborhood around a face
        surface_fea_3 = self.conv_surface_3(verts=verts,
                                            faces=faces,
                                            ring_n=ring_3,
                                            centers=centers)

        if self.include_spectral:
            spectral_fea = self.spectral_descriptor(data["face_feats"])
            fea_in = torch.cat([points_fea, surface_fea_1, surface_fea_2,
                                surface_fea_3, normals_fea, spectral_fea], 1)
        else:
            # Concatenate spatial and structural features
            fea_in = torch.cat([points_fea, surface_fea_1, surface_fea_2, surface_fea_3, normals_fea], 1)

        # Mesh block 1 features
        fea = self.mesh_block_1(fea=fea_in, ring_n=ring_1)

        # Max pool features
        fea = self.max_pool_fea_1(fea=fea, ring_n=ring_1)

        # Randomly select pooling indicies. Face indices not in pooling_idx will not be considered by
        # further layers.
        # Note: pooling_idx is same for all meshes and size of the orginal tensor does not change
        pool_idx = torch.randperm(ring_2.shape[1])[:ring_2.shape[1]//self.pool_rate]

        # Sort the index for correct tensor re-assignment in PsuedoMeshBlock
        pool_idx, _ = torch.sort(pool_idx)

        # Mesh block 2 features
        fea = self.mesh_block_2(fea=fea, ring_n=ring_2, pool_idx=pool_idx)

        # Max pool features
        fea = self.max_pool_fea_2(fea=fea, ring_n=ring_2)

        # Randomly subset pooling indicies from initial pool_idx
        pool_idx_idx = torch.randperm(pool_idx.shape[0])[:pool_idx.shape[0]//self.pool_rate]
        pool_idx = pool_idx[pool_idx_idx]
        pool_idx, _ = torch.sort(pool_idx)

        # Mesh block 3 features
        fea = self.mesh_block_3(fea=fea, ring_n=ring_3, pool_idx=pool_idx)

        # Only consider the pool_idx, global features
        fea = fea[:, :, pool_idx]

        fea = torch.max(fea, dim=2)[0]
        fea = fea.reshape(fea.size(0), -1)
        cls = self.classifier(fea)
        return cls
    


class PsuedoConvFace(torch.nn.Module):
    def __init__(self, in_channel, out_channel, num_neighbor):
        """
        Args:
            in_channel: number of channels in feature

            out_channel: number of channels produced by convolution

            num_neighbor: per faces neighbors in a n-Ring neighborhood.
        """
        super(PsuedoConvFace, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_neighbor = num_neighbor

        self.concat_mlp = torch.nn.Sequential(
            torch.nn.Conv1d(self.in_channel, self.out_channel, 1),
            torch.nn.BatchNorm1d(self.out_channel),
            torch.nn.ReLU(),
        )

    def forward(self, fea, ring_n, pool_idx):
        """
        Args:
            fea: face features of meshes
            [num_meshes, in_channel, num_faces]

            ring_n: faces in a n-Ring neighborhood
            [num_meshes, num_faces, num_neighbor]

            pool_idx: indices of faces to be considered for spatial pooling
            [num_faces]//2 OR [num_faces]//4

        Returns:
            conv_fea: features produced by convolution of faces with its
            n-Ring neighborhood features
            [num_meshes, out_channel, num_faces]
        """
        num_meshes, num_channels, _ = fea.size()
        _, num_faces, _ = ring_n.size()
        # assert ring_n.shape == (num_meshes, num_faces, self.num_neighbor)

        # Gather features at face neighbors only at pool_idx
        fea = fea.unsqueeze(3)
        ring_n = ring_n.unsqueeze(1)
        ring_n = ring_n.expand(num_meshes, num_channels, num_faces, -1)

        neighbor_fea = fea[
            torch.arange(num_meshes)[:, None, None, None],
            torch.arange(num_channels)[None, :, None, None],
            ring_n
        ]
        neighbor_fea = neighbor_fea.squeeze(4)

        # Pool input feature only at pool_idx
        # Pooling here occurs at the spatial dimension
        fea = fea[:, :, pool_idx, :]

        # Concatenate gathered neighbor features to face_feature, and then find the sum
        fea = torch.cat([fea, neighbor_fea], 3)
        # assert fea.shape == (num_meshes, num_channels, num_faces, self.num_neighbor + 1)
        fea = torch.sum(fea, 3)

        conv_fea = self.concat_mlp(fea)
        # assert conv_fea.shape == (num_meshes, self.out_channel, num_faces)

        return conv_fea

class PsuedoConvFaceBlock(torch.nn.Module):
    """
    Multiple PsuedoConvFaceBlock layers create a PsuedoMeshBlock.
    PsuedoConvFaceBlock is comprised of PsuedoConvFace layers.
    First PsuedoConvFace layer convolves on in_channel to produce "128" channels.
    Second PsuedoConvFace convolves these "128" channels to produce "growth factor" channels.
    These features get concatenated to the original input feature to produce
    "in_channel + growth_factor" channels.
    Note: The original mesh dimensions are maintained for gathering the neighbor features but
    the operations get perfomed only on the pooling indices.
    """
    def __init__(self, in_channel, growth_factor, num_neighbor):
        """
        Args:
        in_channel: number of channels in feature

        growth_factor: number of channels to increase in_channel by

        num_neighbor: per faces neighbors in a n-Ring neighborhood.
        """
        super(PsuedoConvFaceBlock, self).__init__()
        self.in_channel = in_channel
        self.growth_factor = growth_factor
        self.num_neighbor = num_neighbor
        self.pconv_face_1 = PsuedoConvFace(in_channel, 128, num_neighbor)
        self.pconv_face_2 = PsuedoConvFace(128, growth_factor, num_neighbor)

    def forward(self, fea, ring_n, pool_idx):
        """
        Args:
            fea: face features of meshes
            [num_meshes, in_channel, num_faces]

            ring_n: faces in a n-Ring neighborhood
            [num_meshes, num_faces, num_neighbor]

            pool_idx: indices of faces to be considered for spatial pooling
            [num_faces]//2 OR [num_faces]//4

        Returns:
            conv_block_fea: features produced by ConvFaceBlock layer
            [num_meshes, in_channel + growth_factor, num_faces]
        """
        fea_copy = fea
        device = fea.device
        num_meshes, num_channels, num_faces = fea.size()
        # assert ring_n.shape == (num_meshes, pool_idx.shape[0], self.num_neighbor)
        # assert fea.shape == (num_meshes, self.in_channel, num_faces)

        n = torch.arange(num_meshes)[:, None, None]
        p = pool_idx[None, None, :]

        # Convolve
        fea = self.pconv_face_1(fea, ring_n, pool_idx)
        # assert fea.shape == (num_meshes, fea.shape[1], pool_idx.shape[0])
        # Create placeholder for tensor re-assignment
        fea_placeholder = torch.zeros((num_meshes, fea.shape[1], num_faces), device=device)
        c = torch.arange(fea.shape[1])[None, :, None]
        # Assign values from fea to fea_placeholder at pooling indicies
        # Values at non pooling indices will be zero
        fea_placeholder[n, c, p] = fea
        # assert fea_placeholder.shape == (num_meshes, fea.shape[1], num_faces)

        # Convolve
        fea = self.pconv_face_2(fea_placeholder, ring_n, pool_idx)
        # Create placeholder for tensor re-assignment
        fea_placeholder = torch.zeros((num_meshes, fea.shape[1], num_faces), device=device)
        c = torch.arange(fea.shape[1])[None, :, None]
        # Assign values from fea to fea_placeholder at pooling indicies
        # Values at non pooling indices will be zero
        fea_placeholder[n, c, p] = fea
        # assert fea_placeholder.shape == (num_meshes, fea.shape[1], num_faces)

        conv_block_fea = torch.cat([fea_copy, fea_placeholder], 1)
        # assert conv_block_fea.shape == (num_meshes, self.in_channel + self.growth_factor, num_faces)

        return conv_block_fea

class PsuedoMeshBlock(torch.nn.ModuleDict):
    """
    Multiple PsuedoMeshBlock layers create MeshNet2.
    PsuedoMeshBlock is comprised of several PsuedoConvFaceBlock layers.
    """
    def __init__(self, in_channel, num_block, growth_factor, num_neighbor):
        """
        in_channel: number of channels in feature

        growth_factor: number of channels a single ConvFaceBlock increase in_channel by

        num_block: number of ConvFaceBlock layers in a single MeshBlock

        num_neighbor: per faces neighbors in a n-Ring neighborhood.
        """
        super(PsuedoMeshBlock, self).__init__()
        for i in range(0, num_block):
            layer = PsuedoConvFaceBlock(in_channel, growth_factor, num_neighbor)
            in_channel += growth_factor
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, fea, ring_n, pool_idx):
        """
        Args:
            fea: face features of meshes
            [num_meshes, in_channel, num_faces]

            ring_n: faces in a n-Ring neighborhood
            [num_meshes, num_faces, num_neighbor]

            pool_idx: indices of faces to be considered for spatial pooling
            [num_faces]//2 OR [num_faces]//4

        Returns:
            fea: features produced by MeshBlock layer
            [num_meshes, in_channel + growth_factor * num_block, num_faces]
        """
        ring_n = ring_n[:, pool_idx, :]
        for _, layer in self.items():
            fea = layer(fea, ring_n, pool_idx)
        return fea