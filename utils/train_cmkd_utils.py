import os 
import time
import torch
import numpy as np

import torch.nn.functional as F
from torch_geometric.data import Batch


def train_model(
    args,
    model,
    optimizer,
    scheduler,
    criterion,
    train_dataloader,
    epoch,
    train_metrics=None,
    return_preds=False,
):
    model.student.train()
    model.student_proj.train()
    
    y_pred = []
    y_true = []
    
    running_loss, running_ce, running_kd, running_feat = 0, 0, 0, 0
    if train_metrics is None:
        running_correct = 0

    for batch, data in enumerate(train_dataloader):
        meshes, images, labels = data
        meshes = Batch.from_data_list(meshes).to(args.device, non_blocking=True)
        meshes.x, meshes.edge_index, meshes.batch = meshes.x.to(args.device, non_blocking=True), meshes.edge_index.to(args.device, non_blocking=True), meshes.batch.to(args.device, non_blocking=True)
        images = torch.stack(images).to(args.device, non_blocking=True)
        labels = torch.tensor(labels, dtype=torch.long).to(args.device, non_blocking=True)
        
        optimizer.zero_grad()
        
        loss, loss_ce, loss_kd, loss_feat, s_logits = model(
            meshes, images, labels
        )

        loss.backward()
        optimizer.step()

        predictions = s_logits.argmax(dim=1)
        running_loss += loss * labels.size(0)
        running_ce += loss_ce * labels.size(0)
        running_kd += loss_kd * labels.size(0)
        running_feat += loss_feat * labels.size(0)

        # Update metrics
        if train_metrics is not None:
            for name, metric in train_metrics.items():
                metric.update(predictions, labels)
        else:
            running_correct += (predictions == labels).sum().item()

        # Store Predictions
        y_true.append(labels.cpu())
        y_pred.append(predictions.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()   
    
    # Train outputs
    epoch_loss = (running_loss / len(train_dataloader.dataset)).item()
    epoch_ce = (running_ce / len(train_dataloader.dataset)).item()
    epoch_kd = (running_kd / len(train_dataloader.dataset)).item()
    epoch_feat = (running_feat / len(train_dataloader.dataset)).item()
    
    epoch_loss = [epoch_loss, epoch_ce, epoch_kd, epoch_feat]
    
    # compute metrics at the end of this epoch
    if train_metrics is not None:
        metrics_dict = train_metrics.compute()
        if args.ddp:
            torch.distributed.all_reduce(metrics_dict)
        epoch_acc = metrics_dict["Accuracy"].item()
    else:
        epoch_acc = running_correct / len(train_dataloader.dataset)

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
    #model.student.eval()
    
    y_pred = []
    y_true = []

    running_loss = 0
    if test_metrics is None:
        running_correct = 0
    
    with torch.no_grad():
        
        for batch, data in enumerate(test_dataloader):
            meshes, images, labels = data
            #meshes = Batch.from_data_list(meshes).to(args.device, non_blocking=True)
            #meshes.x, meshes.edge_index, meshes.batch = meshes.x.to(args.device, non_blocking=True), meshes.edge_index.to(args.device, non_blocking=True), meshes.batch.to(args.device, non_blocking=True)
            images = torch.stack(images).to(args.device, non_blocking=True)
            labels = torch.tensor(labels, dtype=torch.long).to(args.device, non_blocking=True)
            """            
            loss, loss_ce, loss_kd, loss_feat, s_logits = model(
                meshes, images, labels
            )"""
            
            s_logits = model.student(images, return_features=False)
            
            loss = criterion(s_logits, labels)
            
            predictions = s_logits.argmax(dim=1)
            running_loss += loss * labels.size(0)

            # Update metrics
            if test_metrics is not None:
                for name, metric in test_metrics.items():
                    metric.update(predictions, labels)
            else:
                running_correct += (predictions == labels).sum().item()

            # Store Predictions
            y_true.append(labels.cpu())
            y_pred.append(predictions.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    
    # Test outputs
    epoch_loss = (running_loss / len(test_dataloader.dataset)).item()
    
    # compute metrics at the end of this epoch
    if test_metrics is not None:
        metrics_dict = test_metrics.compute()
        if args.ddp:
            torch.distributed.all_reduce(metrics_dict)
        epoch_acc = metrics_dict["Accuracy"].item()
    else:
        epoch_acc = running_correct / len(test_dataloader.dataset)

    if return_preds:
        return epoch_loss, epoch_acc, y_true, y_pred
    else:
        return epoch_loss, epoch_acc


def pred_model(
    args,
    model,
    optimizer,
    scheduler,
    criterion,
    pred_dataloader,
    epoch,
    pred_metrics=None,
    return_preds=False,
    task="val",
):
    model.eval()
    #model.student.eval()
    
    y_pred = []
    y_true = []

    running_loss = 0
    if pred_metrics is None:
        running_correct = 0
    
    with torch.no_grad():
        
        for batch, data in enumerate(pred_dataloader):
            images, labels = data
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
                        
            s_logits = model.student(images, return_features=False)
            
            loss = criterion(s_logits, labels)

            predictions = s_logits.argmax(dim=1)
            running_loss += loss * labels.size(0)

            # Update metrics
            if pred_metrics is not None:
                for name, metric in pred_metrics.items():
                    metric.update(predictions, labels)
            else:
                running_correct += (predictions == labels).sum().item()

            # Store Predictions
            y_true.append(labels.cpu())
            y_pred.append(predictions.cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    
    # Test outputs
    epoch_loss = (running_loss / len(pred_dataloader.dataset)).item()
    
    # compute metrics at the end of this epoch
    if pred_metrics is not None:
        metrics_dict = pred_metrics.compute()
        if args.ddp:
            torch.distributed.all_reduce(metrics_dict)
        epoch_acc = metrics_dict["Accuracy"].item()
    else:
        epoch_acc = running_correct / len(pred_dataloader.dataset)

    if return_preds:
        return epoch_loss, epoch_acc, y_true, y_pred
    else:
        return epoch_loss, epoch_acc
