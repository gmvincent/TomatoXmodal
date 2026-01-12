import comet_ml
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import os
import cv2
import torch
import torch.distributed as dist
import numpy as np
from sklearn.manifold import TSNE
import datetime
from collections import Counter

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
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
        if (args.model_name not in ["svm", "rf", "spiral_net", "mdc_gcn"]) or (mode == "pred"):
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
    for input, label in dataloader:
        if label is not None and len(label) > 0:
            labels.append(label.numpy())
    if labels:  # Only concatenate if labels contain data
        labels = np.concatenate(labels)
    else:
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

"""
def get_attention_map(model, img, single_task=True):
    attention_maps = []

    def hook_fn(module, input, output):
        # Torchvision ViT returns (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) == 2:
            attn_weights = output[1]
        else:
            attn_weights = output
        if attn_weights is not None:
            attention_maps.append(attn_weights.detach().cpu())

    def forward_with_weights(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return original_forward(*args, **kwargs)
    
    def my_forward_wrapper(attn_obj):
        def my_forward(x):
            reshape_back = False

            # Swin blocks may receive [B, H, W, C]
            if x.dim() == 4:
                B, H, W, C = x.shape
                x = x.view(B, H * W, C)
                reshape_back = True
            else:
                B, N, C = x.shape
                H = W = int(N**0.5)  # fallback if not 4D

            qkv = attn_obj.qkv(x).reshape(B, -1, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            head_dim = attn_obj.qkv.weight.shape[1] // attn_obj.num_heads
            scale = head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            
            attn = attn.softmax(dim=-1)
            attn_obj.attn_map = attn  # Save attention

            #attn = attn_obj.attn_dropout(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
            x = attn_obj.proj(x)
            #x = attn_obj.proj_drop(x)

            if reshape_back:
                x = x.view(B, H, W, C)

            return x
        return my_forward
    
    model_to_check = model.model if not single_task else model
    is_swin = False

    # Torchvision ViT
    if hasattr(model_to_check, "encoder") and hasattr(model_to_check.encoder, "layers"):
        last_block = model_to_check.encoder.layers[-1]
        sa_module = last_block.self_attention
        original_forward = sa_module.forward
        
        sa_module.forward = forward_with_weights
        hook_handle = sa_module.register_forward_hook(hook_fn)
    elif hasattr(model_to_check, "blocks"):
        last_block = model_to_check.blocks[-1]
        sa_module = last_block.self_attention
        original_forward = sa_module.forward
        
        sa_module.forward = forward_with_weights
        hook_handle = sa_module.register_forward_hook(hook_fn)
    
    # Torchvision Swin
    elif hasattr(model_to_check, "features"):
        last_block = model_to_check.features[-1][1]
        sa_module = last_block.attn
        
        original_forward = sa_module.forward
        sa_module.forward = my_forward_wrapper(sa_module)
        hook_handle = None
        is_swin = True
    else:
        raise ValueError("Unsupported model structure.")
    
    # Forward pass
    with torch.no_grad():
        _ = model(img)

    # Remove hook
    if hook_handle:
        hook_handle.remove()
    if original_forward:
        sa_module.forward = original_forward

    if not attention_maps and not is_swin:
        raise RuntimeError("No attention maps captured.")


    # Propagate CLS attention
    if is_swin: 
        attn = sa_module.attn_map.detach().cpu()
        attn = attn.mean(dim=1)        # [B*windows, tokens, tokens]
        attn = attn.mean(dim=-1)       # [B*windows, tokens]
        attn = attn.mean(dim=0)        # [tokens]
        num_tokens = attn.size(0)
        size = int(num_tokens ** 0.5)

        if size * size == num_tokens:
            mask = attn.reshape(size, size).numpy()
        else:
            print(f"[WARNING] Swin attention map size ({num_tokens}) is not square — returning flat attention.")
            mask = attn.numpy()
    else:    
        attn = attention_maps[-1]  # [B, num_heads, num_tokens, num_tokens]
        if attn.dim() == 4:
            attn = attn.mean(dim=1)[0] # Average over heads, get first item in batch

        # Propagate CLS attention
        identity = torch.eye(attn.size(-1))
        a = (attn + identity) / 2
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ identity
        for _ in range(1, 1):  # Propagation depth = 1
            result = a @ result

        mask = result[0, 1:]  # CLS token to patches
        num_patches = mask.size(0)
        size = int(num_patches ** 0.5)
        mask = mask.reshape(size, size).numpy()
            
    mask = mask / np.max(mask)

    return mask

def plot_attention_maps(args, experiment, model, dataloader, step, mode, num_images=6):
    model.eval()
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
    
    data_iter = iter(dataloader)
    inputs, labels = next(data_iter)
    
    indices = np.random.choice(inputs.shape[0], size=num_images, replace=False)
    inputs = inputs[indices].to(args.device).float()
    labels = labels[indices].to(args.device).long()
    
    for i in range(num_images):
        rgb_img = inputs[i].detach().cpu().permute(1, 2, 0).numpy()
        img_np = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
        rgb_img = Image.fromarray((img_np * 255).astype(np.uint8))
        
        input_img = inputs[i].unsqueeze(0).to(args.device)
        mask = get_attention_map(model, input_img)
        mask_resized = np.array(
            Image.fromarray((mask * 255).astype(np.uint8)).resize(rgb_img.size, resample=Image.BILINEAR)
        ) / 255.0
        jet_colored_mask = cm.get_cmap('jet')(mask_resized)[..., :3]  # drop alpha
        overlay = (0.55 * np.array(rgb_img) / 255.0 + 0.45 * jet_colored_mask).clip(0, 1)
        
        axes[0][0].set_title("Input Image")
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

        axes[i, 1].imshow(mask, cmap="jet")
        axes[0, 1].set_title("Attention Map")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(overlay)
        axes[0, 2].set_title("Overlay")
        axes[i, 2].axis("off")

    plt.tight_layout()
    experiment.log_figure(figure_name=f"{mode}/attention_maps", figure=fig, step=step)
    plt.close(fig)    
"""
    
#TODO: plot true positives, false positives, false negatives
#TODO: plot segmentation masks (for canopy and lesion segmentation)
