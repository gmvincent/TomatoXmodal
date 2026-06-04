import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.hub import download_url_to_file

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from models.dgnet import DGNet
from models.mdc_gcn import MDC_GCN
from models.meshnet import MeshNet2
from models.pointnet import PointNet2
from models.distillation import Distiller
from models.spiral_classifier import SpiralClassifier
from models.efficientnet_student import EfficientNetStudent

def get_model(args, model_name):
    if model_name == "custom_net":
        model = Simple3DCNN(
            num_classes=args.num_classes,
        )
    elif model_name == "point_net":
        model = PointNet2(
            num_class=args.num_classes,
            in_channel=args.input_channels-3,
            use_normals=True,
        )
    elif model_name == "spiral_net":
        model = SpiralClassifier(
            in_channels=args.input_channels,   # multispectral bands
            channels=[32, 64, 128],            # spiral conv hierarchy
            num_classes=args.num_classes
        )
    elif model_name == "mesh_net":
        model = MeshNet2(
            num_cls=args.num_classes,
            num_faces=int(args.target_faces),
            pool_rate=2,
            include_spectral=False,
        )
    elif model_name == "mdc_gcn":
        model = MDC_GCN(
            in_channels=args.input_channels, # 3 or 7
            num_classes=args.num_classes,
        )
    elif model_name == "dgnet":
        model = DGNet(
            in_channels=args.input_channels, 
            num_classes=args.num_classes,
            include_spectral=True,
        )
    elif model_name == "mesh_clip":
        model = -1
    elif model_name == "xmodal":
        teacher_model = get_model(args, "dgnet")
        
        if args.teacher_weights.startswith(('http://', 'https://')):
            print(f"Downloading weights from {args.teacher_weights}...")
            # Create a local filename from the URL
            local_path = os.path.join("/home/gmvincen/TomatoXmodal/outputs", "temp_teacher_weights.pth")
            if not os.path.exists(local_path):
                download_url_to_file(args.teacher_weights, local_path)
            
            load_from = local_path
        else:
            load_from = args.teacher_weights
        state = torch.load(load_from, map_location="cpu", weights_only=False)

        if "state_dict" in state:
            state = state["state_dict"]

        # Clean DDP prefixes
        clean_state = {k.replace("module.", ""): v for k, v in state.items()}

        teacher_model.load_state_dict(clean_state, strict=False)
        student_model = EfficientNetStudent(num_classes=args.num_classes, pretrained=False, input_channels=3)
        
        model = Distiller(teacher_model, student_model, device=args.device, alpha=args.alpha, beta=args.beta, feat_align=args.feat_align)
        
    elif model_name == "mlp":
        model = TabularNN(input_dim=args.input_channels, num_classes=args.num_classes)
    elif model_name == "svm":
        model = SVC(kernel='rbf', probability=True, random_state=42)
    elif model_name == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    elif model_name.startswith("resnet"):
        model_func = getattr(models, model_name)
        model = model_func(weights="IMAGENET1K_V1" if args.pretrained else None)
        if args.input_channels != 3:
            model.conv1 = nn.Conv2d(args.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
        
    elif model_name.startswith("mobilenet"):
        model_func = getattr(models, model_name)
        model = model_func(weights="IMAGENET1K_V1" if args.pretrained else None)
        if args.input_channels != 3:
            model.features[0][0] = nn.Conv2d(args.input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, args.num_classes)
        
    elif model_name.startswith("efficientnet"):
        model_func = getattr(models, model_name)
        model = model_func(weights="IMAGENET1K_V1" if args.pretrained else None)

        if args.input_channels != 3:
            old_conv = model.features[0][0]

            new_conv = nn.Conv2d(
                args.input_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

            with torch.no_grad():
                if args.input_channels >= 3:
                    new_conv.weight[:, :3] = old_conv.weight
                if args.input_channels > 3:
                    new_conv.weight[:, 3:] = old_conv.weight[:, :1].repeat(1, args.input_channels - 3, 1, 1)

            model.features[0][0] = new_conv

        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, args.num_classes)

        
    elif model_name.startswith("densenet"):
        model_func = getattr(models, model_name)
        model = model_func(weights="IMAGENET1K_V1" if args.pretrained else None)
        if args.input_channels != 3:
            model.features.conv0 = nn.Conv2d(args.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier = nn.Linear(model.classifier.in_features, args.num_classes)
        
    elif model_name.startswith("vgg"):
        model_func = getattr(models, model_name)
        model = model_func(weights="IMAGENET1K_V1" if args.pretrained else None)
        if args.input_channels != 3:
            model.features[0] = nn.Conv2d(args.input_channels, 64, kernel_size=3, stride=1, padding=1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, args.num_classes)

    elif model_name.startswith("vit"):
        model_func = getattr(models, model_name)
        model = model_func(weights="IMAGENET1K_V1" if args.pretrained else None)
        if args.input_channels != 3:
            model.conv_proj = nn.Conv2d(args.input_channels, model.conv_proj.out_channels, kernel_size=16, stride=16)
        model.heads.head = nn.Linear(model.heads.head.in_features, args.num_classes)
         
    elif model_name.startswith("swin"):
        model_func = getattr(models, model_name)
        model = model_func(weights="IMAGENET1K_V1" if args.pretrained else None)
        if args.input_channels != 3:
            model.features[0][0] = nn.Conv2d(args.input_channels, model.features[0][0].out_channels,
                                            kernel_size=4, stride=4, padding=0)
        model.head = nn.Linear(model.head.in_features, args.num_classes)

    else:       
        raise ValueError(f"Unknown model name: {model_name}")

    return model


class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(32)
        
        self.pool_mid  = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool_glob = nn.AdaptiveAvgPool3d((2, 2, 2))
        
        self.fc1 = nn.Linear(32 * 2 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool_mid(F.relu(self.bn1(self.conv1(x))))
        x = self.pool_mid(F.relu(self.bn2(self.conv2(x))))
        
        x = F.relu(self.bn3(self.conv3(x))) 
        x = self.pool_glob(x)           
        
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return self.out(x)

class TabularNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TabularNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)