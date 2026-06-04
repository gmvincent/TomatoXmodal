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

def gather_tensor(args, tensor):   
    # Non-scalar: gather all tensors
    tensors_gather = [torch.zeros_like(tensor) for _ in range(args.world_size)]
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0).detach()