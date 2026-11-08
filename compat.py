"""Translate historical checkpoint metadata to the MIIB module names."""
from omegaconf import OmegaConf


def modernize_config(config):
    raw = OmegaConf.to_container(config, resolve=False) if OmegaConf.is_config(config) else config

    def visit(value):
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, str):
            if value == "CoMM":
                return "MIIB"
            return (value.replace("pl_modules.comm.CoMM", "pl_modules.miib.MIIB")
                    .replace("models.mmfusion.MMFusion", "models.miib.MIIBEncoder")
                    .replace("losses.comm_loss.CoMMLoss", "losses.miib_loss.MIIBLoss"))
        return value

    return OmegaConf.create(visit(raw))
