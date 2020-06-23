import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from sync_batchnorm import *


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

        convert_syncBN(self.model.named_modules())

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


def convert_syncBN(modules):
    # convert BN to syncBN
    for name, module in modules:
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            num_features = module.num_features
            w = module.weight
            b = module.bias
            m = module.momentum
            r_mean = module.running_mean
            r_var = module.running_var
            eps = module.eps
            num_batches_tracked = module.num_batches_tracked
            if isinstance(module, nn.BatchNorm2d):
                module = SynchronizedBatchNorm2d(num_features)
            else:
                module = SynchronizedBatchNorm1d(num_features)
            module.weight = w
            module.bias = b
            module.momentum = m
            module.running_mean = r_mean
            module.running_var = r_var
            module.eps = eps
            module.num_batches_tracked = num_batches_tracked


def get_model(config):
    try:
        f = globals().get(f"{config.model_name}")
        print(f"... Model Info - {config.model_name}")
        print("...", end=" ")
        return f(config)

    except TypeError:
        raise NotImplementedError("model name not matched")
