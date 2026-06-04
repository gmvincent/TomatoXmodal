import comet_ml
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import os
import cv2
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import datetime
from collections import Counter

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
from PIL import Image, ImageFilter

from pytorch_grad_cam import GradCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from sklearn.metrics import confusion_matrix

from thop import profile, clever_format

from utils.metrics import initialize_metrics, log_metrics

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold")

# Initialize CometML Experiment
def create_experiment(args):
    now = datetime.datetime.now()
    
    experiment_name = f"{args.model_name}_{args.dataset_name}_{now.strftime('%y%m%d%H%M')}"

    experiment = Experiment(
        api_key="6XqmAhuJUkx6wPhz0sdCRXwRz",
        project_name=args.cometml_project, 
        workspace="gmvincent",
        auto_param_logging=True,
        auto_metric_logging=True,
        log_env_details=True,
    )

    experiment.set_name(experiment_name)
    
    return experiment

def log_experiment(
    args, experiment, metrics, loss, epoch, y_true, y_pred, dataloader, model, mode="train",
):

    # Ensure only rank 0 logs to CometML
    if args.ddp and dist.get_rank() != 0:
        return  
    
    log_metrics(experiment, metrics, loss, epoch, mode)
    
    # log plots
    if (epoch >= args.epochs - 1) or (epoch % args.print_freq == 0):
 
        plot_confusion_matrix(args, experiment, y_true, y_pred, epoch, mode)
        if (args.model_name not in ["svm", "rf", "spiral_net", "mdc_gcn", "custom_net", "point_net", "mesh_net", "dgnet"]) or (mode == "pred"):
            plot_cam(args, experiment, model, dataloader, step=epoch, mode=mode)
        
    if (epoch >= args.epochs - 1) and (mode == "test"):
        # Log Experiment Specific Args
        for arg, value in vars(args).items():
            experiment.log_parameter(arg, value)

    if mode == "pred":
        plot_predictions(args, experiment, model, dataloader, step=epoch, mode=mode)
        plot_cam(args, experiment, model, dataloader, step=epoch, mode=mode)
        
def log_distill_metrics(args, experiment, loss, loss_ce, loss_kd, loss_feat, step, mode="train"):  

    experiment.log_metric(f"{mode}/loss", loss, step=step)
    experiment.log_metric(f"{mode}/loss_ce", loss_ce, step=step)
    experiment.log_metric(f"{mode}/loss_kd", loss_kd, step=step)
    experiment.log_metric(f"{mode}/loss_feat", loss_feat, step=step)
    
def log_model_weights(args, experiment, model):
    # Log model weights
    if args.ddp:
        model = model.module
    log_model(experiment, model, "final_model")
    
    # Calculate and log the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    experiment.log_parameter("num_parameters", total_params)
    
    # Calculate and log the model size (in MB)
    param_size = sum(p.element_size() * p.numel() for p in model.parameters())
    model_size_mb = param_size / (1024 ** 2)
    experiment.log_parameter("model_size_MB", model_size_mb)
    
    # Calculate and log the number of FLOPs
    #flops, params = profile(model, inputs=(torch.zeros(1, args.input_channels, 64, 64)), verbose=False)
    #flops, params = clever_format([flops, params], "%.3f")

    #experiment.log_parameter("num_flops", flops)

def plot_distribution(args, experiment, dataloader, classes, mode):
    fig, ax = plt.subplots(figsize=(14, 11))
    
    labels = []
    for item in range(len(dataloader.dataset)):
        _, label = dataloader.dataset[item]
        
        if label is not None:
            if isinstance(label, torch.Tensor):
                labels.append(label.item())   # tensor → python int
            else:
                labels.append(label)
    if len(labels) == 0:  # Only concatenate if labels contain data
        print(f"No labels found in {mode} dataloader")
        return
    
    label_counts = Counter(labels)
    freqs = [label_counts.get(i, 0) for i in range(args.num_classes)]

    # Bar plot with class names as x-axis ticks
    ax.bar(range(len(args.classes)), freqs, color="orchid")
    ax.set_xticks(range(len(args.classes)))
    ax.set_xticklabels(args.classes, rotation=90, ha="right")
    ax.set_xlabel('')
    ax.set_ylabel('Frequency')
    
    # Log the plot to CometML
    experiment.log_figure(
        figure_name=f"{mode}/data_distribution", figure=plt.gcf()
    )
    plt.close(fig)
    

def plot_confusion_matrix(args, experiment, y_true, y_pred, step, mode):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    
    cm = confusion_matrix(y_true, y_pred, normalize="true", labels=range(args.num_classes))
    
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
                cm, 
                annot=True, 
                fmt=".2f", 
                cmap="Blues", 
                square=True, 
                cbar=False,
                xticklabels=["Control", "Bacterial Spot", "Septoria Leaf Spot", "Early Blight"],#args.classes,
                yticklabels=["Control", "Bacterial Spot", "Septoria Leaf Spot", "Early Blight"],#args.classes,
                ax=ax,
                annot_kws={"size": 18},
                )
    plt.xlabel("Predicted Labels", fontsize=18)
    plt.ylabel("True Labels", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Log the plot to CometML
    experiment.log_figure(figure_name=f"{mode}/cm", figure=plt.gcf(), step=step)
    plt.close(fig)
    

def plot_cam(args, experiment, model, dataloader, step, mode, num_images=4):
    if args.model_name == "xmodal":
        model = model.student
    elif args.ddp:
        model = model.module
    
    target_layer = get_target_layer(args, args.model_name, model)
    cam = GradCAM(model=model, target_layers=target_layer)
    
    data_iter = iter(dataloader)
    if args.dataset_name == "xmodal_features":
        meshes, inputs, labels = next(data_iter)
        inputs = torch.stack(inputs)
        labels = torch.tensor(labels)
    else:
        inputs, labels = next(data_iter)
    
    N = min(num_images, labels.shape[0])
    fig, axes = plt.subplots(N, 2, figsize=(8, 4 * N))

    indices = np.random.choice(inputs.shape[0], size=N, replace=False)
    indices = torch.tensor(indices, dtype=torch.long)

    inputs = inputs[indices].to(args.device).float()
    labels = labels[indices].to(args.device).long()
    
    targets = [ClassifierOutputTarget(label.item()) for label in labels]
    grayscale_cam = cam(input_tensor=inputs, targets=targets)
    
    if N == 1:
        axes = [axes]  # make sure it's iterable

    for i in range(N):
        if inputs.shape[1] > 3:
            inputs = inputs[:, :3, :, :]
        
        rgb_img = inputs[i].cpu().permute(1, 2, 0).numpy()
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)  # normalize
        visualization = show_cam_on_image(rgb_img, grayscale_cam[i], use_rgb=True)

        axes[i][0].imshow(rgb_img)
        axes[i][0].spines['top'].set_visible(False)
        axes[i][0].spines['right'].set_visible(False)
        axes[i][0].spines['bottom'].set_visible(False)
        axes[i][0].spines['left'].set_visible(False)
        axes[i][0].set_xticks([])
        axes[i][0].set_yticks([])
        
        label_idx = int(labels[i].item())
        label_text = args.classes[label_idx] if isinstance(args.classes[0], str) else str(label_idx)
        axes[i][0].set_ylabel(label_text, fontsize=14, rotation=90, labelpad=40, va='center')
        
        axes[i][1].imshow(visualization)
        axes[i][1].axis("off")
        
        if i == 0:
            axes[i][0].set_title("Input Image")
            axes[i][1].set_title("CAM")
            
    plt.tight_layout()
    experiment.log_figure(figure_name=f"{mode}/cam", figure=fig, step=step)
    plt.close(fig)

def get_target_layer(args, model_name, model):
    model_name = model_name.lower()

    model_targets = {
        "fasterrcnn": lambda m: m.backbone,
        "resnet":     lambda m: m.layer4[-1],
        "vgg":        lambda m: m.features[-1],
        "dense":      lambda m: m.features[-1],
        "mobile":     lambda m: m.features[-1],
        "mnasnet":    lambda m: m.layers[-1],
        "vit":        lambda m: m.encoder.ln if hasattr(m, "encoder") else m.encoder.layers[-1].ln_1,
        "swin":       lambda m: m.features[-1][0].norm1 if hasattr(m, "features") else m.norm,
        "efficient":  lambda m: m.features[-1][0],
        "xmodal":     lambda m: m.backbone.features[-1][0],  
    }

    for key, target_fn in model_targets.items():
        if key in model_name:
            args.target_layer = [target_fn(model)]
            return args.target_layer

    raise ValueError(f"No matching target layer found for model: {model_name}")

def plot_predictions(args, experiment, model, dataloader, step, mode, num_images=4):
    if args.model_name == "xmodal":
        pred_model = model.student.eval()
    else:
        pred_model = model.eval()
        
    for class_idx, class_name in enumerate(args.classes):           
        inputs, labels = next(iter(dataloader))
        
        class_mask = (labels == class_idx).nonzero(as_tuple=True)[0]
        N = min(num_images, len(class_mask))
        if N == 0:
            print(f"No samples of class {class_name} in the current batch.")
            continue
        
        fig, axes = plt.subplots(N, 1, figsize=(4, 4 * N))
        if N == 1:
            axes = [axes]

        # randomly sample num_images from this class
        chosen = np.random.choice(class_mask.cpu(), size=N, replace=False)

        # slice inputs + labels
        imgs = inputs[chosen].to(args.device).float()
        labs = labels[chosen].to(args.device).long()
        if imgs.shape[1] > 3:
            imgs = imgs[:, :3]
            
        with torch.no_grad():
            outputs = pred_model(imgs)
            preds = outputs.argmax(dim=1)
        
        for j in range(N):           
            rgb_img = imgs[j].cpu().permute(1, 2, 0).numpy()
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)  # normalize

            ax = axes[j]
            ax.imshow(rgb_img)
            ax.set_xticks([])
            ax.set_yticks([])

            for side in ["top", "bottom", "left", "right"]:
                ax.spines[side].set_visible(False)
                
            pred_label = args.classes[preds[j].item()] 
            ax.set_ylabel(pred_label, fontsize=14, rotation=90, labelpad=40, va='center')

        plt.tight_layout()
        experiment.log_figure(figure_name=f"{mode}/{class_name}_preds", figure=fig, step=step)
        plt.close(fig)


def plot_cosine_similarity(args, experiment, t_feat, s_feat, step, mode):
    t_feat = F.normalize(t_feat, dim=1)
    s_feat = F.normalize(s_feat, dim=1)

    cos_sim = (t_feat * s_feat).sum(dim=1).detach().cpu()

    fig, ax = plt.subplots(figsize=(8,7))
    
    ax.hist(cos_sim.numpy(), bins=50)
    ax.set_title("Teacher-Student Cosine Similarity")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Frequency")
    
    plt.tight_layout()
    experiment.log_figure(figure_name=f"{mode}/feature_similarity", figure=fig, step=step)
    plt.close(fig)
    

def visualize_feat_space(args, experiment, t_feat, s_feat, labels, step, mode):
    t_feat = t_feat.detach().cpu()
    s_feat = s_feat.detach().cpu()
    labels = labels.detach().cpu()
    
    # Combine
    X = torch.cat([t_feat, s_feat], dim=0)
    y = torch.cat([labels, labels], dim=0)

    # Domain labels (teacher vs student)
    domain = torch.cat([
        torch.zeros(len(t_feat)),  # teacher
        torch.ones(len(s_feat))    # student
    ])

    X_embedded = TSNE(n_components=2, perplexity=30, random_state=args.random_seed).fit_transform(X)

    fig, ax = plt.subplots(figsize=(8,7))
    unique_classes = torch.unique(y)
    num_classes = len(unique_classes)
    cmap = plt.cm.get_cmap("tab10", num_classes)

    # Normalize for consistency
    norm = mcolors.Normalize(vmin=y.min().item(), vmax=y.max().item())
    
    # Teacher
    scatter_teacher = ax.scatter(
        X_embedded[:len(t_feat), 0],
        X_embedded[:len(t_feat), 1],
        c=y[:len(t_feat)],
        cmap=cmap,
        norm=norm,
        marker='o',
        alpha=0.8
    )

    # Student
    scatter_student = ax.scatter(
        X_embedded[len(t_feat):, 0],
        X_embedded[len(t_feat):, 1],
        c=y[len(t_feat):],
        cmap=cmap,
        norm=norm,
        marker='x',
        alpha=0.8
    )
    
    legend_elements = []

    # Domain (marker-based)
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                 label='Teacher',
                                 markerfacecolor='gray', markersize=8))

    legend_elements.append(Line2D([0], [0], marker='x', color='gray',
                                 label='Student',
                                 markersize=8))

    # Class colors
    for c in unique_classes:
        c_int = int(c)
        color = cmap(norm(c_int))
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w',
                   label=f'Class {int(c)}',
                   markerfacecolor=cmap(int(c)),
                   markersize=8)
        )

    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    experiment.log_figure(figure_name=f"{mode}/feature_space", figure=fig, step=step)
    plt.close(fig)
    
def visualize_domain_space(args, experiment, lab_feat, field_feat, lab_labels, field_labels):
    lab_feat = lab_feat.detach().cpu()
    field_feat = field_feat.detach().cpu()
    
    lab_labels = lab_labels.detach().cpu()
    field_labels = field_labels.detach().cpu()
    
    # Combine
    X = torch.cat([lab_feat, field_feat], dim=0)
    y = torch.cat([lab_labels, field_labels], dim=0)

    # Domain labels (lab vs field)
    domain = torch.cat([
        torch.zeros(len(lab_feat)),  # lab
        torch.ones(len(field_feat))    # field
    ])

    # t-SNE
    X_embedded = TSNE(n_components=2, perplexity=30, random_state=args.random_seed).fit_transform(X)

    fig, ax = plt.subplots(figsize=(8,7))
    
    unique_classes = torch.unique(y)
    num_classes = len(unique_classes)
    cmap = plt.cm.get_cmap("tab10", num_classes)

    # Normalize for consistency
    norm = mcolors.Normalize(vmin=y.min().item(), vmax=y.max().item())
    
    # Teacher
    scatter_teacher = ax.scatter(
        X_embedded[:len(lab_feat), 0],
        X_embedded[:len(lab_feat), 1],
        c=y[:len(lab_feat)],
        cmap=cmap,
        norm=norm,
        marker='o',
        alpha=0.8
    )

    # Student
    scatter_student = ax.scatter(
        X_embedded[len(lab_feat):, 0],
        X_embedded[len(lab_feat):, 1],
        c=y[len(lab_feat):],
        cmap=cmap,
        norm=norm,
        marker='x',
        alpha=0.8
    )
    
    legend_elements = []

    # Domain (marker-based)
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                 label='Lab Domain',
                                 markerfacecolor='gray', markersize=8))

    legend_elements.append(Line2D([0], [0], marker='x', color='gray',
                                 label='Field Domain',
                                 markersize=8))

    # Class colors
    for c in unique_classes:
        c_int = int(c)
        color = cmap(norm(c_int))

        legend_elements.append(
            Line2D([0], [0], marker='o', color='w',
                   label=f'Class {c_int}',
                   markerfacecolor=color,
                   markersize=8)
        )

    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    experiment.log_figure(figure_name=f"domain_space", figure=fig)
    plt.close(fig)