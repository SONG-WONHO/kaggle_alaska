import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import pretrainedmodels


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.eff = EfficientNet.from_pretrained('efficientnet-b0')

        def get_reg_layer(c=1280):
            return nn.Sequential(
                nn.Linear(c, config.num_targets),
            )

        self.c_list = [
            16,
            24, 24,
            40, 40,
            80, 80, 80,
            112, 112, 112,
            192, 192, 192, 192,
            320,
            1280
        ]

        self.dense_out = [get_reg_layer(c) for i, c in enumerate(self.c_list)]

    def forward(self, x):
        batch_size = x.size(0)

        # Stem
        results = []
        x = self.eff._swish(self.eff._bn0(self.eff._conv_stem(x)))

        # Blocks
        for idx, block in enumerate(self.eff._blocks):
            drop_connect_rate = self.eff._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.eff._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            results.append(F.avg_pool2d(x, x.size()[2:]).reshape(batch_size, -1))

        # Head
        feat = self.eff._swish(self.eff._bn1(self.eff._conv_head(x)))
        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)
        results.append(feat)

        # LLFs
        results = torch.cat([self.dense_out[i](results[i]) for i in range(len(self.c_list))])

        return results


def get_model(config):
    try:
        f = globals().get(f"{config.model_name}")
        print(f"... Model Info - {config.model_name}")
        print("...", end=" ")
        return f(config)

    except TypeError:
        raise NotImplementedError("model name not matched")
