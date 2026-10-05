import torch
import torchmetrics
import torchmetrics.regression
import torch.distributed as dist

import numpy as np

class PredictionTime(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_time", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, start_time, end_time, batch_size: int):
        duration = torch.tensor(end_time - start_time, dtype=torch.float32, device=self.device)
        self.total_time += duration
        self.count += batch_size

    def compute(self):
        if self.count == 0:
            return torch.tensor(0.0, device=self.device)
        return self.total_time / self.count

    def reset(self):
        super().reset()

def _build_classification_metrics(num_classes):
    return {
        "Accuracy": torchmetrics.Accuracy(num_classes=num_classes, task="multiclass"),
        "F1Score": torchmetrics.F1Score(average=None, num_classes=num_classes, task="multiclass"),
        "Recall": torchmetrics.Recall(average=None, num_classes=num_classes, task="multiclass"),  # also called Sensitivity
        "Precision": torchmetrics.Precision(average=None, num_classes=num_classes, task="multiclass"),
        "Specificity": torchmetrics.Specificity(average=None, num_classes=num_classes, task="multiclass"),
        "F1_macro": torchmetrics.F1Score(average="macro", num_classes=num_classes, task="multiclass"),
        "MCC": torchmetrics.MatthewsCorrCoef(num_classes=num_classes, task="multiclass"),
        #"PredictionTime": PredictionTime(),
    }
 
def _build_regression_metrics(num_outputs):
    return {
        "MAE": torchmetrics.regression.MeanAbsoluteError(),
        "MSE": torchmetrics.regression.MeanSquaredError(),
        "RMSE": torchmetrics.regression.MeanSquaredError(squared=False),
        "R2": torchmetrics.regression.R2Score(), #num_outputs=num_outputs),
        #"PearsonCorrCoef": torchmetrics.regression.PearsonCorrCoef(), #num_outputs=num_outputs),
        #"PredictionTime": PredictionTime(),
    }

def build_metrics(args, task, num_classes):
    _METRIC_BUILDERS = {
        "classification": _build_classification_metrics,
        "regression": _build_regression_metrics,
    }
    
    if task not in _METRIC_BUILDERS:
        raise ValueError(f"Unsupported task '{task}'. Use one of: {sorted(_METRIC_BUILDERS)}.")
    metrics = _METRIC_BUILDERS[task](num_classes)

    for metric in metrics.values():
        metric.to(args.device)
    return torchmetrics.MetricCollection(metrics)

def initialize_metrics(args):       
    train_metrics = build_metrics(args, args.task, args.num_classes)
    val_metrics, test_metrics = train_metrics.clone(), train_metrics.clone()
    return train_metrics, val_metrics, test_metrics

def log_metrics(args, experiment, metrics, loss, step, mode="train"):
    
    def _log(key, val, task_classes):
        val = val.cpu().detach()
        if val.ndim > 0 and val.numel() > 1:
            for i, v_ in enumerate(val):
                experiment.log_metric(f"{key}_{task_classes[i]}", v_.item(), step=step)
        else:
            experiment.log_metric(key, val.item(), step=step)
            
    if metrics is not None:
        for name, value in metrics.items():
            _log(f"{mode}/{name}", value.compute(), args.classes)

    # Log loss (shared across tasks or single)
    experiment.log_metric(f"{mode}/loss", loss if loss is not None else 0, step=step)

def gather_tensor(args, tensor):   
    # Non-scalar: gather all tensors
    tensors_gather = [torch.zeros_like(tensor) for _ in range(args.world_size)]
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0).detach()