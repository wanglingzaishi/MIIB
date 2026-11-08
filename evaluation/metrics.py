"""Dataset-level supervised metrics, with explicit conversion from logits."""
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, MultilabelAccuracy, F1Score
from torchmetrics.regression import MeanSquaredError, PearsonCorrCoef, R2Score


def supervised_metrics(task, num_outputs):
    if task == "binary_head":
        return nn.ModuleDict({"acc1": BinaryAccuracy(), "f1": BinaryF1Score()})
    if task == "bce_with_logits":
        return nn.ModuleDict({
            "acc1": MultilabelAccuracy(num_labels=num_outputs, average="macro"),
            "f1_mean": F1Score(task="multilabel", num_labels=num_outputs, average="macro"),
            "f1_weighted": F1Score(task="multilabel", num_labels=num_outputs, average="weighted"),
        })
    if task == "mse":
        return nn.ModuleDict({
            "mse": MeanSquaredError(num_outputs=num_outputs),
            "pearson_r": PearsonCorrCoef(num_outputs=num_outputs),
            "r2": R2Score(num_outputs=num_outputs, multioutput="uniform_average"),
        })
    return nn.ModuleDict()


def update_metrics(metrics, logits, targets, task):
    classification = task in {"binary_head", "bce_with_logits"}
    predictions = logits.sigmoid() if classification else logits
    targets = targets.int() if classification else targets
    for metric in metrics.values():
        metric.update(predictions, targets)
