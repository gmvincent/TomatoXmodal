import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from data_utils.voxel_dataset import VoxelDataset
from data_utils.voxel_dataset import classes as voxel_classes
from data_utils.voxel_dataset import augment_voxel

from data_utils.morphology_data import MorphologyDataset
from data_utils.morphology_data import classes as morphology_classes

from data_utils.sideview_dataset import TomatoSideViewDataset
from data_utils.sideview_dataset import classes as sideview_classes

from data_utils.mesh_data import MeshDataset
from data_utils.mesh_data import classes as mesh_classes
from data_utils.mesh_data import mesh_to_graph

from data_utils.field_images import FieldTomatoImages
from data_utils.field_images import classes as field_classes

from data_utils.distillation_data import XModalDataset
from data_utils.distillation_data import classes as xmodal_classes

MESH_DATASETS = {
    "meshes": "mesh",
    "spirals": "spiral",
    "pointclouds": "pcd",
    "graph_meshes": "graph",
}

def create_data_loader(args, rank): 
    # Set transforms    
    RGB_MEAN, RGB_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    img_size = (224, 224)
    if args.dataset_name == "rgbd_images":
        normalize = transforms.Normalize(RGB_MEAN + [0.0], RGB_STD + [1.0])
    else:
        normalize = transforms.Normalize(RGB_MEAN, RGB_STD)

    train_transforms = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.Resize(img_size),
    ]

    val_transforms = [transforms.Resize(img_size)]

    # Only append normalize if the dataset requires it
    #if args.dataset_name in ["rgbd_images", "rgb_images", "field_images"]:#, "xmodal_features"]:
    #    train_transforms.append(normalize)
    #    val_transforms.append(normalize)

    if args.dataset_name == "voxel_images":
        train_augmentations = None #augment_voxel
        val_augmentations = None
    else:
        train_augmentations = transforms.Compose(train_transforms)
        val_augmentations = transforms.Compose(val_transforms)
    
    train_ds, val_ds, test_ds, classes = get_datasets(args, rank, args.dataset_name, train_augmentations, val_augmentations)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=args.world_size, rank=rank, shuffle=True) if args.ddp and args.dataset_name!="field_images" else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, num_replicas=args.world_size, rank=rank, shuffle=False) if args.ddp and args.dataset_name!="field_images" else None
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds, num_replicas=args.world_size, rank=rank, shuffle=False) if args.ddp else None
    
    num_workers = 4
    base_loader_args = {
        "batch_size": 32 if args.dataset_name =="field_images" else args.batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": True if num_workers > 0 else False,
    }
    train_loader_args = {**base_loader_args, "drop_last": False} 
    #True if args.ddp else False}
    eval_loader_args = {**base_loader_args, "drop_last": False}
    
    # Create data loaders
    if args.dataset_name == "graph_meshes":
        from torch_geometric.loader import DataLoader as GDataLoader
        train_loader = GDataLoader(
            dataset=train_ds,
            sampler=train_sampler, 
            shuffle=(train_sampler is None),
            **train_loader_args,
            )
        
        val_loader = GDataLoader(
            dataset=val_ds,
            sampler=val_sampler,
            shuffle=False,
            **eval_loader_args,
        )
        
        test_loader = GDataLoader(
            dataset=test_ds,
            sampler=test_sampler,
            shuffle=False,
            **eval_loader_args,            
        )
        return train_loader, val_loader, test_loader, classes
    elif args.dataset_name == "field_images":
        test_loader = torch.utils.data.DataLoader(
            dataset=test_ds,
            sampler=test_sampler,
            shuffle=(test_sampler is None),
            **eval_loader_args,     
        )
        return None, None, test_loader, classes
    else:
        if args.dataset_name == "meshes":
            collate_fn = lambda x: x
        elif args.dataset_name == "xmodal_features":
            collate_fn = lambda batch: tuple(zip(*batch))
        else:
            from torch.utils.data.dataloader import default_collate
            collate_fn = None #default_collate
        
        train_loader = torch.utils.data.DataLoader(
            dataset=train_ds,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=collate_fn,
            **train_loader_args, 
        )
        
        val_loader = torch.utils.data.DataLoader(
            dataset=val_ds,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=collate_fn,
            **eval_loader_args, 
        )
        
        test_loader = torch.utils.data.DataLoader(
            dataset=test_ds,
            sampler=test_sampler,
            shuffle=False,
            collate_fn=collate_fn,
            **eval_loader_args, 
        )

        return train_loader, val_loader, test_loader, classes
    
def get_datasets(args, rank, dataset_name, train_augs, val_augs):
    
    if dataset_name.lower() == "voxel_images":
        
        data_path = "cmwilli5_drive/gmvincen_data/tomato_diseases/scaled_voxel_data"

        full_ds = VoxelDataset(
                    root=os.path.join(args.root, data_path),
                    transform=val_augs,
                )
        
        total_size = len(full_ds)
        if rank==0: print(f"Total Dataset Size: {total_size}")
        indices = list(range(total_size))
        train_indices, val_test_indices = train_test_split(indices, test_size=0.3, random_state=args.random_seed)
        val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, random_state=args.random_seed)

        train_ds = torch.utils.data.Subset(
            VoxelDataset(
                    root=os.path.join(args.root, data_path),
                    transform=train_augs,
                ), 
            train_indices)
        
        val_ds = torch.utils.data.Subset(full_ds, val_indices)
        
        test_ds = torch.utils.data.Subset(full_ds, test_indices)
        
        classes = voxel_classes
    elif dataset_name.lower() in MESH_DATASETS:
        
        data_path = "cmwilli5_drive/gmvincen_data/tomato_diseases/cleaned_phenospex_polygons"
        
        representation = MESH_DATASETS[dataset_name.lower()]
        
        full_ds = MeshDataset(
                    root=os.path.join(args.root, data_path),
                    target_faces=args.target_faces,
                    representation=representation,
                )
        
        total_size = len(full_ds)
        if rank==0:print(f"Total Dataset Size: {total_size}")
        indices = list(range(total_size))
        
        train_indices, val_test_indices = train_test_split(indices, test_size=0.3, random_state=args.random_seed)
        val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, random_state=args.random_seed)

        train_ds = torch.utils.data.Subset(full_ds, train_indices)
        
        val_ds = torch.utils.data.Subset(full_ds, val_indices)
        
        test_ds = torch.utils.data.Subset(full_ds, test_indices)
        
        
        classes = mesh_classes
    elif dataset_name.lower() == "xmodal_features":
        data_path = "cmwilli5_drive/gmvincen_data/tomato_diseases/cleaned_phenospex_polygons"

        full_ds = XModalDataset(
                    root=os.path.join(args.root, data_path), 
                    transform=val_augs,
                    representation="graph",
                    target_faces=args.target_faces,
                )
        
        total_size = len(full_ds)
        if rank==0:print(f"Total Dataset Size: {total_size}")
        indices = list(range(total_size))
        
        train_indices, val_test_indices = train_test_split(indices, test_size=0.3, random_state=args.random_seed)
        val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, random_state=args.random_seed)

        train_ds = torch.utils.data.Subset(
            XModalDataset(
                    root=os.path.join(args.root, data_path),
                    transform=train_augs,
                    graph=True,
                    target_faces=args.target_faces,
                ), 
            train_indices)
                
        val_ds = torch.utils.data.Subset(full_ds, val_indices)
        
        test_ds = torch.utils.data.Subset(full_ds, test_indices)
        
        classes = xmodal_classes
    
    elif dataset_name.lower() == "morphology_features":
        
        data_path = "/home/gmvincen/cmwilli5_drive/gmvincen_data/tomato_diseases/"
        full_ds = MorphologyDataset(
                    root=os.path.join(args.root, data_path),
                    transform=None,
                )
        
        total_size = len(full_ds)
        indices = list(range(total_size))
        train_indices, val_test_indices = train_test_split(indices, test_size=0.3, random_state=args.random_seed)
        val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, random_state=args.random_seed)

        train_ds = torch.utils.data.Subset(full_ds, train_indices)
        
        val_ds = torch.utils.data.Subset(full_ds, val_indices)
        
        test_ds = torch.utils.data.Subset(full_ds, test_indices)
        
        classes = morphology_classes
        
    elif dataset_name.lower() == "rgb_images":
        data_path = "/home/gmvincen/cmwilli5_drive/gmvincen_data/tomato_diseases/cropped_rgbd"
        
        full_ds = TomatoSideViewDataset(
                    root=os.path.join(args.root, data_path),
                    transform=val_augs,
                    depth=False,
                )
        
        total_size = len(full_ds)
        if rank==0:print(f"Total Dataset Size: {total_size}")

        indices = list(range(total_size))
        train_indices, val_test_indices = train_test_split(indices, test_size=0.3, random_state=args.random_seed)
        val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, random_state=args.random_seed)

        train_ds = torch.utils.data.Subset(
            TomatoSideViewDataset(
                    root=os.path.join(args.root, data_path),
                    transform=train_augs,
                    depth=False
                ), 
            train_indices)
        
        val_ds = torch.utils.data.Subset(full_ds, val_indices)
        
        test_ds = torch.utils.data.Subset(full_ds, test_indices)
        
        classes = sideview_classes
        
    elif dataset_name.lower() == "rgbd_images":
        data_path = "/home/gmvincen/cmwilli5_drive/gmvincen_data/tomato_diseases/cropped_rgbd"
        full_ds = TomatoSideViewDataset(
                    root=os.path.join(args.root, data_path),
                    transform=val_augs,
                    depth=True,
                )
        
        total_size = len(full_ds)
        indices = list(range(total_size))
        train_indices, val_test_indices = train_test_split(indices, test_size=0.3, random_state=args.random_seed)
        val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, random_state=args.random_seed)

        train_ds = torch.utils.data.Subset(
            TomatoSideViewDataset(
                    root=os.path.join(args.root, data_path),
                    transform=train_augs,
                    depth=True
                ), 
            train_indices)
        
        val_ds = torch.utils.data.Subset(full_ds, val_indices)
        
        test_ds = torch.utils.data.Subset(full_ds, test_indices)
        
        classes = sideview_classes
        
    elif dataset_name.lower() == "field_images":
        data_path = "/home/gmvincen/cmwilli5_drive/gmvincen_data/tomato_diseases/field_images"
        test_ds = FieldTomatoImages(
                    root=os.path.join(args.root, data_path),
                    transform=val_augs,
                )
        
        train_ds = None
        val_ds = None
        
        classes = field_classes
        
    return train_ds, val_ds, test_ds, classes