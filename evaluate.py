"""Evaluate an SSL checkpoint using its saved architecture and probe settings."""
import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from compat import modernize_config
from run import execute


def checkpoint_config(path):
    checkpoint = torch.load(path, map_location="cpu")
    if "hyper_parameters" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError("Expected a Lightning MIIB checkpoint with hyper_parameters and state_dict")
    cfg = modernize_config(checkpoint["hyper_parameters"])
    del checkpoint
    if cfg.model.name != "MIIB":
        raise ValueError("evaluate.py expects an SSL checkpoint; use finetune.py for a supervised checkpoint")
    target = cfg.data.data_module._target_
    family = "trifeatures" if "Trifeatures" in target else ("mmimdb" if "MMIMDB" in target else "multibench")
    cfg.family = family
    dataset = cfg.data.data_module.get("dataset", cfg.data.name)
    names = {"sarcasm": "mustard", "humor": "urfunny2", "visionandtouch": "vt_ee", "visionandtouch-bin": "vt_contact2"}
    cfg.experiment = cfg.get("experiment", names.get(dataset, dataset))
    if family == "trifeatures":
        cfg.experiment = "trifeatures_synergy" if cfg.data.data_module.get("biased", False) else "trifeatures_ru"
        cfg.linear_probing = {"_target_": "evaluation.linear_probe.LinearProbingCallback", "val_loaders": False}
    if family == "mmimdb" and os.environ.get("MIIB_TEXT_MODEL"):
        cfg.mmimdb.encoders[1].model_name = os.environ["MIIB_TEXT_MODEL"]
    cfg.mode = "test"
    cfg.ckpt_path = str(Path(path).resolve())
    cfg.probe = True
    cfg.save_checkpoints = False
    cfg.early_stopping = False
    cfg.threads = 4
    cfg.evaluation_protocol = "historical"
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("--workers", type=int, help="Override saved worker count; this can change evaluation randomness")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("--checkpoint does not exist")
    cfg = checkpoint_config(args.checkpoint)
    cfg.trainer.accelerator = args.device
    cfg.trainer.strategy = "auto"
    cfg.trainer.devices = 1
    cfg.trainer.num_nodes = 1
    cfg.trainer.default_root_dir = str(args.output)
    cfg.trainer.enable_progress_bar = False
    cfg.trainer.enable_model_summary = False
    if args.workers is not None:
        cfg.data.data_module.num_workers = args.workers
    output = execute(cfg)
    print("Metrics and resolved configuration: " + str(output))


if __name__ == "__main__":
    main()
