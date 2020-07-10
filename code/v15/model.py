import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), x.size()[2:]).pow(1./self.p)


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = EfficientNet.from_pretrained(config.backbone_name)

        self.c = {
            "efficientnet-b0": 1280,
            "efficientnet-b3": 1536
        }[config.backbone_name]

        def get_reg_layer():
            return nn.Sequential(
                nn.Linear(self.c, config.num_targets),
            )

        self.pool_layer = GeM()

        self.dense_out = get_reg_layer()

    def forward(self, x):
        feat = self.model.extract_features(x)
        # feat = self.pool_layer(feat).reshape(-1, self.c)
        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, self.c)
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
