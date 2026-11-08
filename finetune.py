"""Fine-tune a supplied MIIB checkpoint using the original supervised heads."""
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from run import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="mosi")
    parser.add_argument("--checkpoint", type=Path, required=True, help="MIIB SSL checkpoint used to initialize the encoder")
    parser.add_argument("--test-checkpoint", type=Path, help="Fine-tuned checkpoint to evaluate")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--no-checkpoints", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("--checkpoint does not exist")
    cfg = OmegaConf.load(Path(__file__).resolve().parent / "configs/experiments" / (args.experiment + ".yaml"))
    regression = args.experiment == "vt_ee"
    multilabel = cfg.family == "mmimdb"
    head = "mse" if regression else ("bce_with_logits" if multilabel else "binary_head")
    cfg.model = {"name": "SupervisedClassifier", "model": {
        "_target_": "pl_modules.supervised.SupervisedClassifier", "model": head,
        "ckpt_path": str(args.checkpoint.resolve()), "num_classes": 4 if regression else (23 if multilabel else 2),
        "head_lr": args.head_lr, "p": args.dropout}}
    cfg.optim = {"lr": args.lr, "weight_decay": 0.01}
    cfg.trainer.max_epochs = args.epochs
    cfg.trainer.accelerator = args.device
    cfg.trainer.default_root_dir = args.output
    cfg.experiment = "finetune_" + args.experiment
    cfg.save_checkpoints = not args.no_checkpoints
    cfg.data.data_module.num_workers = 0
    if args.test_checkpoint:
        cfg.mode = "test"
        cfg.ckpt_path = str(args.test_checkpoint.resolve())
    execute(cfg)


if __name__ == "__main__":
    main()
