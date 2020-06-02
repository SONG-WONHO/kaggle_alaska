import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        model = EfficientNet.from_pretrained('efficientnet-b0')
        model = [m for m in list(model.children()) if not isinstance(m, nn.BatchNorm2d)]
        modules = []
        for block in list(model[1].children()):
            modules.append(
                nn.Sequential(*[m for m in list(block.children()) if not isinstance(m, nn.BatchNorm2d)]))
        modules = nn.Sequential(*modules)
        model[1] = modules
        model = nn.Sequential(*model[:3])
        self.model = model

        def get_reg_layer():
            return nn.Sequential(
                nn.Linear(1280, config.num_targets),
            )

        self.dense_out = get_reg_layer()

    def forward(self, x):
        feat = self.model(x)
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
