import os
import torch

import comet_ml
from comet_ml.integration.pytorch import log_model


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, save_best_model=True):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_best_model (bool): True if model weights should be saved for the best model.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.save_best_model = save_best_model
        self.best_weights = None

    def __call__(self, val_loss, model, epoch, experiment):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.save_best_model:
                self.best_weights = model.state_dict()
                #log_model(experiment, model, f"best_model_epoch{epoch}")
        elif val_loss >= self.best_loss - self.min_delta:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement in validation loss
            self.best_loss = val_loss
            self.counter = 0
            if self.save_best_model:
                self.best_weights = model.state_dict()
                #log_model(experiment, model, f"best_model_epoch{epoch}")
