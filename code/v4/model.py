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
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), x.size()[2:]).pow(1./self.p).reshape(-1, 1280)


class LinearPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1, 1280, 256), requires_grad=True)

    def forward(self, x):
        x = x.reshape(-1, 1280, 16 * 16) * self.p
        return x.mean(dim=-1)


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = EfficientNet.from_pretrained('efficientnet-b0')

        # self.pool_layer = GeM()
        self.pool_layer = LinearPool()

        def get_reg_layer():
            return nn.Sequential(
                nn.Dropout(0.15),
                nn.Linear(1280, config.num_targets),
            )

        self.dense_out = get_reg_layer()

    def forward(self, x):
        feat = self.model.extract_features(x)
        feat = self.pool_layer(feat)
        outputs = self.dense_out(feat)
        return outputs


class GridModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gs = 16
        self.config = config

        self.model = EfficientNet.from_pretrained('efficientnet-b0')

        self.pool_layer = LinearPool()

        def get_reg_layer():
            return nn.Sequential(
                nn.Dropout(0.15),
                nn.Linear(1280, config.num_targets),
            )

        self.dense_out = get_reg_layer()

    def forward(self, x):
        bs, c, w, h = x.size()
        gs = self.gs
        w_t, h_t = 16, 16

        x = x.reshape(bs, c, gs, int(w//gs), gs, int(h//gs))
        x = x.permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(bs, c, gs * gs, int(w//gs), int(h//gs))
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(bs * gs * gs, c, int(w//gs), int(h//gs))

        feat = self.model.extract_features(x)
        feat = feat.reshape(bs, gs * gs, 1280, int(w_t//gs), int(h_t//gs))
        feat = feat.permute(0, 2, 1, 3, 4)
        feat = feat.reshape(bs, 1280, gs, gs, int(w_t//gs), int(h_t//gs))
        feat = feat.permute(0, 1, 2, 4, 3, 5)
        feat = feat.reshape(bs, 1280, gs * int(w_t//gs), gs * int(h_t//gs))

        feat = self.pool_layer(feat)
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
