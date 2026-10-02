import os 
import time
import torch
import numpy as np

from torch_geometric.data import Batch as PyGBatch

def train_model(
    args,
    model,
    experiment,
    optimizer,
    scheduler,
    criterion,
    train_dataloader,
    epoch,
    train_metrics=None,
    return_preds=False,
):
    model.train()

    y_pred, y_true = [], []
    running_loss, running_samples = 0, 0
    if train_metrics is None:
        running_correct = 0
    
    for data in train_dataloader:
        inputs, labels = prepare_batch(args, data)

        optimizer.zero_grad()
        output = model(*inputs)
        
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()

        preds = output.argmax(dim=1)
        running_loss += loss.item() * labels.size(0)
        running_samples += labels.size(0)
        
        # Update metrics
        if train_metrics is not None:
            train_metrics.update(output, labels)
        else:
            running_correct += (preds == labels).sum().item()

        # Store Predictions
        y_true.append(labels)
        y_pred.append(preds)
    
    y_true = torch.cat(y_true).detach()
    y_pred = torch.cat(y_pred).detach()  
    
    # Train outputs
    epoch_loss = running_loss / running_samples
    
    # compute metrics at the end of this epoch
    if train_metrics is not None:
        metrics_dict = train_metrics.compute()
        epoch_acc = metrics_dict["Accuracy"].item()
    else:
        epoch_acc = running_correct / running_samples
        
    if return_preds:
        return epoch_loss, epoch_acc, y_true, y_pred
    else:
        return epoch_loss, epoch_acc

def test_model(
    args,
    model,
    optimizer,
    scheduler,
    criterion,
    test_dataloader,
    epoch,
    test_metrics=None,
    return_preds=False,
    task="val",
):
    model.eval()

    y_pred, y_true = [], []
    running_loss, running_samples = 0, 0
    if test_metrics is None:
        running_correct = 0

    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = prepare_batch(args, data)

            output = model(*inputs)
            
            loss = criterion(output, labels) 

            preds = output.argmax(dim=1)
            running_loss += loss.item() * labels.size(0)
            running_samples += labels.size(0)
            
            # Update metrics
            if test_metrics is not None:
                test_metrics.update(output, labels)
            else:
                running_correct += (preds == labels).sum().item()
            """
            mis_idx = (preds != labels)

            probs = torch.softmax(output, dim=1)
            mis_probs = probs[mis_idx]
            mis_preds = preds[mis_idx]
            mis_labels = labels[mis_idx]
            
            top2 = torch.topk(mis_probs, k=2, dim=1)

            for i in range(min(10, len(mis_probs))):
                print(
                    f"True: {mis_labels[i].item()} | "
                    f"Pred: {mis_preds[i].item()} | "
                    f"Top2: {top2.indices[i].tolist()} | "
                    f"Conf: {top2.values[i].tolist()}"
                )
            """                
            # Store Predictions
            y_true.append(labels)
            y_pred.append(preds)

    y_true = torch.cat(y_true).detach()
    y_pred = torch.cat(y_pred).detach()   

    # Test outputs
    epoch_loss = running_loss / running_samples

    # compute metrics at the end of this epoch
    if test_metrics is not None:
        metrics_dict = test_metrics.compute()
        epoch_acc = metrics_dict["Accuracy"].item()
    else:
        epoch_acc = running_correct / running_samples
        
    if return_preds and task=="val":
        return epoch_loss, epoch_acc, y_true, y_pred
    elif return_preds and task=="predict":
        return epoch_loss, epoch_acc, y_true, y_pred, None
    else:
        return epoch_loss, epoch_acc


def prepare_batch(args, data):
    """Return (model_inputs_tuple, labels) on args.device for any dataset type."""
    # Graphs (MDC_GCN): PyG Batch object
    if isinstance(data, PyGBatch):
        data = data.to(args.device, non_blocking=True)
        return (data.x, data.edge_index, data.batch), data.y

    # Spirals (SpiralNet++): dict from spiral_collate, batch size 1
    if args.dataset_name == "spirals":
        x = data["features"][0].to(args.device, non_blocking=True)
        s = [si.to(args.device, non_blocking=True) for si in data["spiral_indices"]]
        d = [dt.to(args.device, non_blocking=True) for dt in data["down_transform"]]
        labels = data["label"].to(args.device, dtype=torch.long)
        return (x, s, d), labels
    
    instances, labels = data[0], data[1]
    labels = labels.to(args.device, non_blocking=True, dtype=torch.long)

    # Meshes (MeshNet++, DGNet): list of dicts -> one batched dict
    if args.dataset_name == "meshes":
        instances = [{k: v for k, v in m.items() if k != "vert_feats"} for m in instances]
        max_verts = max(m["verts"].shape[0] for m in instances)

        batched = {}
        for k in instances[0].keys():
            if k == "verts":
                padded = torch.zeros(len(instances), max_verts, 3)
                for i, m in enumerate(instances):
                    padded[i, :m["verts"].shape[0]] = m["verts"]
                batched[k] = padded.to(args.device, non_blocking=True)
            else:
                batched[k] = torch.stack([m[k] for m in instances]).to(args.device, non_blocking=True)
        return (batched,), labels
    
    # Point clouds (PointNet++, DGCNN): dense [B, C, N] tensor or Images
    instances = instances.to(args.device, non_blocking=True, dtype=torch.float)
    return (instances,), labels