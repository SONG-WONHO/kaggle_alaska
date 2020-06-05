import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import pretrainedmodels
import math


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(1280, 320, 2, stride=2),
            nn.BatchNorm2d(320),
            nn.ConvTranspose2d(320, 192, 2, stride=2),
            nn.BatchNorm2d(192),
            nn.ConvTranspose2d(192, 80, 2, stride=2),
            nn.BatchNorm2d(80),
            nn.ConvTranspose2d(80, 40, 2, stride=2),
            nn.BatchNorm2d(40),
            nn.ConvTranspose2d(40, 3, 2, stride=2),)

        def get_reg_layer():
            return nn.Sequential(
                nn.Linear(1280, config.num_targets),
            )

        self.dense_out = get_reg_layer()

    def forward(self, x, pretrain=False):
        feat = self.model.extract_features(x)

        if pretrain:
            feat = self.upsampler(feat)
            return feat

        else:
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
