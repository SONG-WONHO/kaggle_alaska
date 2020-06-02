import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.header = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 3, (3, 3), padding=1),
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 3, (1, 1)),
            nn.BatchNorm2d(3)
        )

        self.model = EfficientNet.from_pretrained('efficientnet-b0')

        def get_reg_layer():
            return nn.Sequential(
                nn.Linear(1280, config.num_targets),
            )

        self.dense_out = get_reg_layer()

    def forward(self, x):
        x = self.header(x)
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
