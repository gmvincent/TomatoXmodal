import torch
import torchmetrics
import torchmetrics.regression

import numpy as np

class PredictionTime(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        # Default tensors are now explicitly on the metric's device
        self.add_state("total_time", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, total_time):
        device = self.total_time.device 
        self.total_time += torch.tensor(total_time, device=device)
        self.count += torch.tensor(1, device=device)

    def compute(self):
        return self.total_time / self.count

    def reset(self):
        device = self.total_time.device
        self.total_time = torch.tensor(0.0, device=device)
        self.count = torch.tensor(0, device=device)


def initialize_metrics(args):
    def build_metrics(num_classes):
        metrics = {
            "Accuracy": torchmetrics.Accuracy(num_classes=num_classes, task="multiclass"),
            #"F1": torchmetrics.F1Score(average="none", num_classes=num_classes, task="multiclass"),
            "Recall_macro": torchmetrics.Recall(average="macro", num_classes=num_classes, task="multiclass"), # also called Sensitivity
            "Precision_macro": torchmetrics.Precision(average="macro", num_classes=num_classes, task="multiclass"),
            "Specificity_macro": torchmetrics.Specificity(average="macro", num_classes=num_classes, task="multiclass"),
            "F1_macro": torchmetrics.F1Score(average="macro", num_classes=num_classes, task="multiclass"),
            "MCC": torchmetrics.MatthewsCorrCoef(num_classes=num_classes, task="multiclass"),
            #"PredictionTime": PredictionTime(),
        }

        for metric in metrics.values():
            metric.to(args.device)
        return torchmetrics.MetricCollection(metrics)
        
    train_metrics = build_metrics(args.num_classes)
    val_metrics, test_metrics = train_metrics.clone(), train_metrics.clone()
    return train_metrics, val_metrics, test_metrics
    
    


def log_metrics(experiment, metrics, loss, step, mode="train"):
    for name, value in metrics.items():
        val = value.compute()
        experiment.log_metric(
            f"{mode}/{name}", val.cpu().detach().numpy().tolist(), step=step
        )

    # Log loss (shared across tasks or single)
    experiment.log_metric(f"{mode}/loss", loss if loss is not None else 0, step=step)
