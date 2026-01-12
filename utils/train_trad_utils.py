import os
import time
import torch

import numpy as np

def train_model(
    args,
    model,
    train_dataloader,
    epoch,
    train_metrics=None,
    return_preds=False,
):

    X_train, y_train = [], []
    
    for batch in train_dataloader:
        X_batch, y_batch = batch

        # Move data to CPU if necessary
        X_batch = X_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()

        # Accumulate the averaged data
        X_train.append(X_batch)
        y_train.append(y_batch)

    # Concatenate all the batches into one array
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    
    # Fit model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
                                 
    # Train outputs
    epoch_loss = None
    epoch_acc = None
    
    # compute metrics at the end of this epoch
    if train_metrics is not None:
        for name, metric in train_metrics.items():
            metric.update(torch.tensor(y_pred).cpu(), torch.tensor(y_train).cpu())
        metrics_dict = train_metrics.compute()
        epoch_acc = metrics_dict["Accuracy"].item()
    

    if return_preds:
        return epoch_loss, epoch_acc, y_train, y_pred
    else:
        return epoch_loss, epoch_acc

def test_model(
    args,
    model,
    test_dataloader,
    epoch,
    test_metrics=None,
    return_preds=False,
    task="val",
):

    X_test, y_test = [], []

    for batch in test_dataloader:
        X_batch, y_batch = batch

        # Move data to CPU if necessary
        X_batch = X_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()

        # Accumulate the data
        X_test.append(X_batch)
        y_test.append(y_batch)
    
    # Concatenate all the batches into one array
    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)
        
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
                                 
    # Train outputs
    epoch_loss = None
    epoch_acc = None
    
    # compute metrics at the end of this epoch
    if test_metrics is not None:
        for name, metric in test_metrics.items():
            metric.update(torch.tensor(y_pred).cpu(), torch.tensor(y_test).cpu())
        metrics_dict = test_metrics.compute()
        epoch_acc = metrics_dict["Accuracy"].item()
        
    if ((epoch >= args.epochs - 1) or (epoch % args.print_freq == 0)) and (task == "val"):
        print("\nVal Accuracy: %.5f" % (epoch_acc))


    if return_preds:
        return epoch_loss, epoch_acc, y_test, y_pred
    else:
        return epoch_loss, epoch_acc