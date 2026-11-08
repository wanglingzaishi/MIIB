"""Translate historical checkpoint metadata to the MIIB module names."""
from omegaconf import OmegaConf

#: Renamed ``MIIBLoss`` hyper-parameters: old name -> new name.
LOSS_KWARG_RENAMES = {
    "penalty": "synergy_uniformity_weight",
    "lambda_fusion_penalty": "fusion_uniformity_weight",
    "lambda_single_penalty": "unimodal_uniformity_weight",
    "margin": "diversity_margin",
    "penalty_l": "synergy_weight",
}

#: Hyper-parameters that the old signature accepted but never used.
LOSS_KWARG_DROPPED = ("penalty_fs",)


def _rename_loss_kwargs(model_spec):
    """Rename the ``loss_kwargs`` of a model specification in place-safe fashion."""
    if not isinstance(model_spec, dict):
        return model_spec
    loss_kwargs = model_spec.get("loss_kwargs")
    if not isinstance(loss_kwargs, dict):
        return model_spec
    renamed = {}
    for key, value in loss_kwargs.items():
        if key in LOSS_KWARG_DROPPED:
            continue
        renamed[LOSS_KWARG_RENAMES.get(key, key)] = value
    model_spec["loss_kwargs"] = renamed
    return model_spec


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

    raw = visit(raw)

    # Checkpoints trained before the loss-hyperparameter renaming store the old
    # names under ``model.model.loss_kwargs``; translate them so that such
    # checkpoints stay loadable.
    if isinstance(raw, dict):
        model = raw.get("model")
        if isinstance(model, dict):
            _rename_loss_kwargs(model.get("model"))

    return OmegaConf.create(raw)
