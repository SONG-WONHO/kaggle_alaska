import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = EfficientNet.from_pretrained('efficientnet-b0')

        def get_reg_layer():
            return nn.Sequential(
                nn.Linear(1280, 1280),
                nn.LayerNorm(1280),
                nn.LeakyReLU(),
                nn.Linear(1280, config.num_targets),
            )

        self.dense_out = get_reg_layer()

    def forward(self, x, typ):
        feat = self.model.extract_features(x)
        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)
        outputs = self.dense_out(feat)
        return outputs


def get_model(config):
    try:
        f = globals().get(f"{config.model_name}")
        print(f"... Model Info - {config.model_name}")
        print("...", end=" ")
        return f(config)

    except TypeError:
        raise NotImplementedError("model name not matched")
