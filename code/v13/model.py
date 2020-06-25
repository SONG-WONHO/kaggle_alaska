import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from sync_batchnorm import convert_model


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = EfficientNet.from_pretrained(config.backbone_name)

        print("... Convert BN to SyncBN")
        self.model = convert_model(self.model)

        self.c = {
            'efficientnet-b0': 1280,
            'efficientnet-b1': 1280,
            'efficientnet-b2': 1408,
            'efficientnet-b3': 1536,
            'efficientnet-b4': 1792,
            'efficientnet-b5': 2048,
            'efficientnet-b6': 2304,
            'efficientnet-b7': 2560}[config.backbone_name]

        def get_reg_layer():
            return nn.Sequential(
                nn.Linear(self.c, config.num_targets),
            )

        self.dense_out = get_reg_layer()

    def forward(self, x):
        feat = self.model.extract_features(x)
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
