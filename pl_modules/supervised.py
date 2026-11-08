from pl_modules.base import BaseModel
from typing import Dict
import torch.nn as nn
import sys
import math
import torch
import torch.nn.functional as F
from losses.supcon import SupConLoss
from utils import LinearWarmupCosineAnnealingLR
from evaluation.metrics import supervised_metrics, update_metrics


class SupervisedClassifier(BaseModel):
    """
        Supervised contrastive classifier model on top of a pre-trained feature extractor.
    """

    def __init__(self, ckpt_path: str,
                 num_classes: int,
                 optim_kwargs: Dict,
                 model: str = "supcon",
                 unsupervised_weight: float = 1.0,
                 head_lr: float = None,
                 p: float = 0.1):

        super().__init__(optim_kwargs)
        assert model in {"supcon", "bce_with_logits", "mse", "binary_head"}
        self.ckpt_path = ckpt_path
        self.num_classes = num_classes
        self.base_encoder, embed_dim = self._parse_ckpt(self.ckpt_path)
        self.model = model
        self.unsupervised_weight = float(unsupervised_weight)
        self.head_lr = None if head_lr is None else float(head_lr)
        self.p = float(p)
        if self.model == "supcon":
            self.loss = SupConLoss()
        elif self.model in {"binary_head", "bce_with_logits"}:
            num_labels = 1 if self.model == "binary_head" else num_classes
            self.loss = nn.BCEWithLogitsLoss()
            self.linear = nn.Sequential(
                nn.Dropout(p=self.p),
                nn.Linear(embed_dim,num_labels)
            )
        elif self.model == "mse":
            self.loss = nn.MSELoss()
            self.linear = nn.Linear(embed_dim, num_classes)
        else:
            raise NotImplementedError()

        self.metrics = supervised_metrics(self.model, num_classes)

    @staticmethod
    def _param_groups(named_params, weight_decay: float, lr: float):
        p_wd, p_non_wd = [], []
        for n, p in named_params:
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
        groups = []
        if p_wd:
            groups.append({"params": p_wd, "weight_decay": weight_decay, "lr": lr})
        if p_non_wd:
            groups.append({"params": p_non_wd, "weight_decay": 0, "lr": lr})
        return groups

    def configure_optimizers(self):
        if self.head_lr is None or not hasattr(self, "linear"):
            return super().configure_optimizers()

        base_lr = self.optim_kwargs["lr"]
        head_lr = self.head_lr
        if head_lr == base_lr:
            return super().configure_optimizers()

        weight_decay = self.optim_kwargs["weight_decay"]
        base_params, head_params = [], []
        for n, p in self.named_parameters():
            if n.startswith("linear."):
                head_params.append((n, p))
            else:
                base_params.append((n, p))
        optim_params = (
            self._param_groups(base_params, weight_decay, base_lr) +
            self._param_groups(head_params, weight_decay, head_lr)
        )
        optimizer = torch.optim.AdamW(optim_params)

        if "lr_scheduler" in self.optim_kwargs:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.optim_kwargs["lr_scheduler"]["warmup_epochs"],
                max_epochs=self.trainer.max_epochs,
                warmup_start_lr=self.optim_kwargs["lr_scheduler"]["start_warmup_value"],
                eta_min=self.optim_kwargs["lr_scheduler"]["final_value"]
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        return optimizer

    def _parse_ckpt(self, pth: str):
        from hydra.utils import instantiate
        ckpt = torch.load(pth, map_location="cpu")
        from compat import modernize_config
        hparams = modernize_config(ckpt["hyper_parameters"])
        dataset = hparams["data"]["data_module"].get("dataset", hparams["data"]["name"])
        model_name = hparams["model"]["name"]
        kwargs = dict()
        if model_name == "MIIB":
            specification = hparams["model"] if "encoders" in hparams["model"] else hparams[dataset]
            encoders = instantiate(specification["encoders"])
            adapters = instantiate(specification["adapters"])
            kwargs["encoder"] = {
                "encoders": encoders,
                "input_adapters": adapters}
            model = instantiate(hparams["model"]["model"], optim_kwargs=hparams["optim"], **kwargs)
            embed_dim = hparams["model"]["model"]["encoder"]["embed_dim"]
            out = model.load_state_dict(ckpt["state_dict"])
            print(out)
        else:
            raise NotImplementedError()
        return model, embed_dim

    def _encode_features(self, X):
        if isinstance(X, torch.Tensor):
            X = [X]
        
        output = self.base_encoder.encoder(X)
        if isinstance(output, tuple):
            output = output[0]

        if isinstance(output, list):
            if len(output) != 1:
                raise ValueError(f"Expected one feature tensor, got {len(output)} tensors.")
            output = output[0]

        return output

    def forward(self, X):
        if self.model == "supcon":
            X1, X2 = X
            embed1 = self._encode_features(X1)
            embed2 = self._encode_features(X2)
            return embed1, embed2
        elif self.model in ["binary_head", "bce_with_logits", "mse"]:
            embed = self._encode_features(X)
            out = self.linear(embed)
            return out
        raise NotImplementedError()

    def _format_targets_for_loss(self, y_pred, y):
        y = y.to(y_pred.device)
        if self.model not in {"binary_head", "bce_with_logits"}:
            return y.float()
        if y.shape == y_pred.shape:
            return y.float()
        if y_pred.ndim == 2 and y_pred.size(1) == 1:
            return y.float().view(-1, 1)
        if y_pred.ndim == 2 and y.dim() == 1:
            return F.one_hot(y.long(), num_classes=y_pred.size(1)).float()
        if y_pred.ndim == 2 and y.dim() == 2 and y.size(1) == 1:
            return F.one_hot(y.long().view(-1), num_classes=y_pred.size(1)).float()
        return y.float()

    def training_step(self, batch, batch_idx):
        log_dict = None
        if self.model == "supcon":
            if len(batch) == 4:
                X1, X2, y, label_mask = batch
            else:
                X1, X2, y = batch
                label_mask = None
            Z1, Z2 = self.forward((X1, X2))
            Z = torch.stack((Z1, Z2), dim=1)
            if label_mask is None:
                loss = self.loss(Z, y)
            else:
                label_mask = label_mask.to(device=Z.device, dtype=torch.bool).view(-1)
                unsup_loss = self.loss(Z)
                labeled_count = int(label_mask.sum().item())
                if labeled_count > 0:
                    sup_loss = self.loss(Z[label_mask], y[label_mask])
                else:
                    sup_loss = Z.sum() * 0.0
                loss = sup_loss + self.unsupervised_weight * unsup_loss
                log_dict = {
                    "loss": loss,
                    "sup_loss": sup_loss,
                    "unsup_loss": unsup_loss,
                    "labeled_batch_ratio": label_mask.float().mean()
                }
        else:
            X, y = batch
            y_pred = self.forward(X)
            targets = self._format_targets_for_loss(y_pred, y)
            loss = self.loss(y_pred, targets)
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)
        if log_dict is None:
            log_dict = {"loss": loss}
        self.log_dict(log_dict, on_epoch=True,
                      sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.model == "supcon":
            if len(batch) == 4:
                X1, X2, y, _ = batch
            else:
                X1, X2, y = batch
            Z1, Z2 = self.forward((X1, X2))
            Z = torch.stack((Z1, Z2), dim=1)
            val_loss = self.loss(Z, y)
        else:
            X, y = batch
            y_pred = self.forward(X)
            targets = self._format_targets_for_loss(y_pred, y)
            val_loss = self.loss(y_pred, targets)
            update_metrics(self.metrics, y_pred, targets, self.model)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        if self.model == "supcon":
            if len(batch) == 4:
                X1, X2, y, _ = batch
            else:
                X1, X2, y = batch
            Z1, Z2 = self.forward((X1, X2))
            Z = torch.stack((Z1, Z2), dim=1)
            test_loss = self.loss(Z, y)
        else:
            X, y = batch
            y_pred = self.forward(X)
            targets = self._format_targets_for_loss(y_pred, y)
            test_loss = self.loss(y_pred, targets)
            update_metrics(self.metrics, y_pred, targets, self.model)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, sync_dist=True)
        return test_loss

    def _reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def _log_epoch_metrics(self):
        for name, metric in self.metrics.items():
            self.log(name, metric.compute().mean(), on_step=False, on_epoch=True,
                     sync_dist=True, prog_bar=True)

    def on_validation_epoch_start(self):
        self._reset_metrics()

    def on_test_epoch_start(self):
        self._reset_metrics()

    def on_validation_epoch_end(self):
        self._log_epoch_metrics()

    def on_test_epoch_end(self):
        self._log_epoch_metrics()

    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs):
        return self.base_encoder.extract_features(loader, **kwargs)
