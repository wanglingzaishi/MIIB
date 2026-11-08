import timm
import torch
import torch.nn as nn
import os


class VisionTransformer(nn.Module):
    """Pre-trained vision transformers supported by timm library. """

    def __init__(self, model_name: str,
                 pretrained: bool = True,
                 freeze: bool = False,
                 output_value: str = "embedding",
                 checkpoint_path: str = None):
        super().__init__()
        assert output_value in {'embedding', "token_embeddings"}

        checkpoint_path = checkpoint_path or os.environ.get("MIIB_VISION_CHECKPOINT")
        options = {"pretrained_cfg_overlay": {"file": checkpoint_path}} if checkpoint_path else {}
        if output_value == "token_embeddings":
            self.model = timm.create_model(model_name, global_pool="", pretrained=pretrained, **options)
            self.model.head = nn.Identity() # get token embeddings
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, **options)
        self.model_name = model_name
        self.freeze = freeze
        self.pretrained = pretrained

        if freeze: # no grad computed
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        if self.freeze:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)
