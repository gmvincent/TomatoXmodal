import os 
import time
import torch
import numpy as np

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
    model.train()

    y_pred = []
    y_true = []
    
    running_loss = 0
    if train_metrics is None:
        running_correct = 0
    
    for batch, data in enumerate(train_dataloader):      
        instances, labels = data
        instances = instances.to(args.device, non_blocking=True, dtype=torch.float)
        labels = labels.to(args.device, non_blocking=True, dtype=torch.long)    

        optimizer.zero_grad()
        
        output = model(instances)
        
        #torch.cuda.synchronize()
        
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()

        predictions = output
        
        running_loss += loss * labels.size(0)

        # Update metrics
        if train_metrics is not None:
            for name, metric in train_metrics.items():
                metric.update(predictions, labels)
        else:
            preds = predictions.argmax(dim=1)
            running_correct += (preds == labels).sum().item()

        # Store Predictions
        y_true.append(labels)
        y_pred.append(predictions.argmax(dim=1))
    
    
    y_true = torch.cat(y_true).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()   
    
    # Train outputs
    epoch_loss = (running_loss / len(train_dataloader.dataset)).item()
    
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

    y_pred = []
    y_true = []

    running_loss = 0
    if test_metrics is None:
        running_correct = 0
    
    with torch.no_grad():
        for batch, data in enumerate(test_dataloader):           
            instances, labels = data
            instances = instances.to(args.device, non_blocking=True, dtype=torch.float)
            labels = labels.to(args.device, non_blocking=True, dtype=torch.long)   

            output = model(instances)
            
            #torch.cuda.synchronize()
            
            loss = criterion(output, labels)

            predictions = output
            running_loss += loss * labels.size(0)

            # Update metrics
            if test_metrics is not None:
                for name, metric in test_metrics.items():
                    metric.update(predictions, labels)
            else:
                preds = predictions.argmax(dim=1)
                running_correct += (preds == labels).sum()

            # Store Predictions
            y_true.append(labels)
            y_pred.append(predictions.argmax(dim=1))

    y_true = torch.cat(y_true).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).detach().cpu().numpy()   

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
