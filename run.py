"""Train/evaluate the extracted MIIB implementation with portable experiment configs."""
import hashlib
import importlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


def build_model(cfg):
    if cfg.model.name == "SupervisedClassifier":
        return instantiate(cfg.model.model, optim_kwargs=cfg.optim)
    enc = cfg.model if cfg.family == "trifeatures" else cfg[dataset_name(cfg)]
    return instantiate(
        cfg.model.model, optim_kwargs=cfg.optim,
        encoder={"encoders": instantiate(enc.encoders), "input_adapters": instantiate(enc.adapters)},
    )


def dataset_name(cfg):
    if cfg.family == "trifeatures":
        return "bimodal_trifeatures"
    if cfg.family == "mmimdb":
        return "mmimdb"
    return cfg.data.data_module.dataset


def build_data(cfg, model=None, **extra):
    model = cfg.model.name if model is None else model
    if cfg.family == "multibench":
        spec = cfg[dataset_name(cfg)]
        kwargs = dict(modalities=spec.modalities, task=spec.task)
        if model in {cfg.model.name, "SupervisedClassifier"}:
            kwargs.update(OmegaConf.to_container(spec.kwargs))
        kwargs.update(extra)
        return instantiate(cfg.data.data_module, model=model, **kwargs)
    return instantiate(cfg.data.data_module, model=model, **extra)


class EpochEvidence(Callback):
    """Small numerical evidence; no tensor/checkpoint copies."""
    def __init__(self, output):
        self.output = Path(output)
        self.records = []

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        row = {k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v)
               for k, v in trainer.callback_metrics.items()}
        row.update(epoch=trainer.current_epoch, step=trainer.global_step)
        self.records.append(row)
        print("MIIB_EPOCH " + json.dumps(row, sort_keys=True), flush=True)

    def on_fit_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        digest = hashlib.sha256()
        for name, tensor in pl_module.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        evidence = {"epochs": self.records, "state_sha256": digest.hexdigest(),
                    "global_step": trainer.global_step, "device": str(pl_module.device)}
        (self.output / "evidence.json").write_text(json.dumps(evidence, indent=2))


def execute(cfg):
    devices = cfg.trainer.get("devices", 1)
    device_count = len(devices) if isinstance(devices, (list, tuple)) or OmegaConf.is_list(devices) else devices
    if device_count != 1 or cfg.trainer.get("num_nodes", 1) != 1:
        raise ValueError("The archived probe protocol supports one device per run. Run independent seeds on separate GPUs.")
    if cfg.get("evaluation_protocol", "historical") != "historical":
        raise ValueError("Only the documented historical evaluation protocol is implemented")
    torch.set_num_threads(int(cfg.threads))
    # Eager imports before seeding also make source-vs-extraction comparisons fair.
    importlib.import_module(cfg.model.model._target_.rsplit(".", 1)[0])
    import models.transformer
    import dataset.multibench
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    model = build_model(cfg)
    model.save_hyperparameters(cfg)
    supervised = cfg.model.name == "SupervisedClassifier"
    data_mode = ("Sup" if cfg.model.model.get("model") in {"binary_head", "bce_with_logits", "mse"}
                 else "SupervisedClassifier") if supervised else cfg.model.name
    data = build_data(cfg, data_mode)
    root = Path(cfg.trainer.default_root_dir) / "MIIB" / cfg.experiment / ("seed_%s" % cfg.seed)
    logger = CSVLogger(str(root), name="csv_logs")
    _ = logger.experiment
    output = Path(logger.log_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    callbacks = []
    if cfg.probe and not (supervised and data_mode == "Sup"):
        if cfg.family == "trifeatures":
            names = ["share", "unique1", "unique2", "synergy"]
            downstream = [build_data(cfg, "Sup", biased=False, task=t) for t in names]
        else:
            names, downstream = [dataset_name(cfg)], [build_data(cfg, "Sup")]
        callbacks.append(instantiate(cfg.linear_probing, downstream_data_modules=downstream, names=names))
    if supervised and data_mode == "Sup":
        monitor = "mse" if cfg.model.model.model == "mse" else ("f1_mean" if cfg.family == "mmimdb" else "acc1")
    elif cfg.probe:
        regression = "LinearProbingRegCallback" in cfg.linear_probing._target_
        monitor = "mse" if regression else ("f1_mean" if cfg.family == "mmimdb" else "acc1")
    else:
        monitor = "val_loss"
    direction = "min" if monitor in {"mse", "val_loss"} else "max"
    if cfg.save_checkpoints:
        callbacks.append(ModelCheckpoint(dirpath=str(output / "checkpoints"), monitor=monitor,
                                         mode=direction, save_top_k=1, auto_insert_metric_name=False,
                                         filename="epoch-{epoch:02d}-step-{step}"))
    if cfg.early_stopping:
        callbacks.append(EarlyStopping(monitor=monitor, mode=direction, patience=15))
    callbacks.append(EpochEvidence(output))
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks,
                          enable_checkpointing=bool(cfg.save_checkpoints))
    if cfg.mode == "train":
        trainer.fit(model, datamodule=data, ckpt_path=cfg.ckpt_path)
    elif cfg.mode == "test":
        if not cfg.ckpt_path:
            raise ValueError("mode=test requires ckpt_path=<trained checkpoint>")
        trainer.test(model, datamodule=data, ckpt_path=cfg.ckpt_path)
    else:
        raise ValueError("mode must be train or test")
    summary = {"mode": cfg.mode, "experiment": cfg.experiment, "seed": int(cfg.seed),
               "evaluation_protocol": cfg.get("evaluation_protocol", "historical"),
               "metrics": {name: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
                           for name, value in trainer.callback_metrics.items()}}
    (output / "result.json").write_text(json.dumps(summary, indent=2))
    return output


@hydra.main(version_base=None, config_path="configs/experiments", config_name="mosi")
def main(cfg: DictConfig):
    execute(cfg)


if __name__ == "__main__":
    main()
