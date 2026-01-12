import os
import random
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
#import torchio as tio
import torchvision.transforms as transforms
from torch.utils.data._utils.collate import default_collate

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

def create_data_loader(args):
    # set random seed
    torch.manual_seed(args.random_seed)
    
    # Set transforms
    #train_augmentations = augment_voxel
    #val_augmentations = None
    
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
    if args.dataset_name in ["rgbd_images", "rgb_images", "field_images", "xmodal_features"]:
        train_transforms.append(normalize)
        val_transforms.append(normalize)

    train_augmentations = transforms.Compose(train_transforms)
    val_augmentations = transforms.Compose(val_transforms)
    
    train_ds, val_ds, test_ds, classes = get_datasets(args, args.dataset_name, train_augmentations, val_augmentations)
    
    
    # Create data loaders
    if args.dataset_name == "graph_meshes":
        train_loader = DataLoader(
            dataset=train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            )
        
        val_loader = DataLoader(
            dataset=val_ds,
            batch_size=args.batch_size,
            shuffle=False,
        )
        
        test_loader = DataLoader(
            dataset=test_ds,
            batch_size=args.batch_size,
            shuffle=False,
        )
        return train_loader, val_loader, test_loader, classes
    elif args.dataset_name == "field_images":
        test_loader = DataLoader(
            dataset=test_ds,
            batch_size=32,#args.batch_size,
            shuffle=True,
            num_workers=4, 
            pin_memory=True,
        )
        return None, None, test_loader, classes
    
    elif args.dataset_name == "xmodal_features":
        train_loader = torch.utils.data.DataLoader(
            dataset=train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,   
            collate_fn=lambda batch: tuple(zip(*batch)), 
            pin_memory=True, 
        )
        
        val_loader = torch.utils.data.DataLoader(
            dataset=val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,   
            collate_fn=lambda batch: tuple(zip(*batch)), 
            pin_memory=True,   
        )
        
        test_loader = torch.utils.data.DataLoader(
            dataset=test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,   
            collate_fn=lambda batch: tuple(zip(*batch)), 
            pin_memory=True,   
        )
        return train_loader, val_loader, test_loader, classes
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_ds,
            batch_size=args.batch_size,
            num_workers=8,
            drop_last=False,
            persistent_workers=True,
            shuffle=True,
            pin_memory=True,
            collate_fn=(lambda x: x) if args.dataset_name == "meshes" else default_collate,
        )
        
        val_loader = torch.utils.data.DataLoader(
            dataset=val_ds,
            batch_size=args.batch_size,
            num_workers=8,
            drop_last=False,
            persistent_workers=True,
            shuffle=False,
            pin_memory=True,
            collate_fn=(lambda x: x) if args.dataset_name == "meshes" else default_collate,
        )
        
        test_loader = torch.utils.data.DataLoader(
            dataset=test_ds,
            num_workers=8,
            batch_size=args.batch_size,
            drop_last=False,
            persistent_workers=True,
            shuffle=False,
            pin_memory=True,
            collate_fn=(lambda x: x) if args.dataset_name == "meshes" else default_collate,
        )

        return train_loader, val_loader, test_loader, classes
    
def get_datasets(args, dataset_name, train_augs, val_augs):
    
    if dataset_name.lower() == "voxel_images":
        
        data_path = "cmwilli5_drive/gmvincen_data/tomato_diseases/scaled_voxel_data"

        full_ds = VoxelDataset(
                    root=os.path.join(args.root, data_path),
                    transform=val_augs,
                )
        
        total_size = len(full_ds)
        print(f"Total Dataset Size: {total_size}")
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
    elif dataset_name.lower() == "meshes":
        
        data_path = "cmwilli5_drive/gmvincen_data/tomato_diseases/phenospex_polygons"

        full_ds = MeshDataset(
                    root=os.path.join(args.root, data_path),
                )
        
        total_size = len(full_ds)
        print(f"Total Dataset Size: {total_size}")
        indices = list(range(total_size))
        
        train_indices, val_test_indices = train_test_split(indices, test_size=0.3, random_state=args.random_seed)
        val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, random_state=args.random_seed)

        train_ds = torch.utils.data.Subset(full_ds, train_indices)
        
        val_ds = torch.utils.data.Subset(full_ds, val_indices)
        
        test_ds = torch.utils.data.Subset(full_ds, test_indices)
        
        
        classes = mesh_classes
    
    elif dataset_name.lower() == "graph_meshes":
        
        data_path = "cmwilli5_drive/gmvincen_data/tomato_diseases/phenospex_polygons"

        full_ds = MeshDataset(
                    root=os.path.join(args.root, data_path), 
                    graph=True,
                )
        
        total_size = len(full_ds)
        print(f"Total Dataset Size: {total_size}")
        indices = list(range(total_size))
        
        train_indices, val_test_indices = train_test_split(indices, test_size=0.3, random_state=args.random_seed)
        val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, random_state=args.random_seed)

        train_ds = torch.utils.data.Subset(full_ds, train_indices)
        
        val_ds = torch.utils.data.Subset(full_ds, val_indices)
        
        test_ds = torch.utils.data.Subset(full_ds, test_indices)
        
        
        classes = mesh_classes
    elif dataset_name.lower() == "xmodal_features":
        data_path = "cmwilli5_drive/gmvincen_data/tomato_diseases/phenospex_polygons"

        full_ds = XModalDataset(
                    root=os.path.join(args.root, data_path), 
                    transform=val_augs,
                )
        
        total_size = len(full_ds)
        print(f"Total Dataset Size: {total_size}")
        indices = list(range(total_size))
        
        train_indices, val_test_indices = train_test_split(indices, test_size=0.3, random_state=args.random_seed)
        val_indices, test_indices = train_test_split(val_test_indices, test_size=0.5, random_state=args.random_seed)

        train_ds = torch.utils.data.Subset(
            XModalDataset(
                    root=os.path.join(args.root, data_path),
                    transform=train_augs,
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
        print(f"Total Dataset Size: {total_size}")

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