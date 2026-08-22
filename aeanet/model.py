import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath, trunc_normal_

from .modules import (
    BottomUpAttentionPropagation,
    ContextGuidedFeaturePyramid,
    EfficientBilinearPooling,
)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class AEANet(nn.Module):
    def __init__(
        self,
        num_classes,
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        attention_channels=16,
        drop_path_rate=0.2,
        layer_scale_init_value=1e-6,
        head_init_scale=0.001,
    ):
        super().__init__()
        if attention_channels < 3:
            raise ValueError("attention_channels must be at least 3")
        self.attention_channels = attention_channels

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )
        )
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        offset = 0
        for i in range(4):
            self.stages.append(
                nn.Sequential(
                    *[
                        ConvNeXtBlock(
                            dims[i], rates[offset + j], layer_scale_init_value
                        )
                        for j in range(depths[i])
                    ]
                )
            )
            offset += depths[i]

        self.feature_pyramid = ContextGuidedFeaturePyramid(
            dims[1], dims[2], dims[3], attention_channels
        )
        self.attention_propagation = BottomUpAttentionPropagation(attention_channels)
        self.ebp = EfficientBilinearPooling()
        self.head = nn.Linear(dims[-1] * attention_channels, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward_backbone(self, x):
        outputs = []
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
            outputs.append(x)
        return outputs

    def forward(self, x, return_attention=True):
        stage1, stage2, stage3, stage4 = self.forward_backbone(x)
        p1, p2, p3 = self.feature_pyramid(stage2, stage3, stage4)
        enhanced_p3, attention_weights = self.attention_propagation(p1, p2, p3)
        response_maps = F.relu(enhanced_p3, inplace=False)
        bilinear_feature = self.ebp(stage4, enhanced_p3)
        logits = self.head(bilinear_feature)
        if return_attention:
            return logits, response_maps
        return logits, None

    def load_convnext_pretrained(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint.get("model", checkpoint)
        state_dict = dict(state_dict)
        for key in ("head.weight", "head.bias"):
            state_dict.pop(key, None)
        state_dict.pop("norm.weight", None)
        state_dict.pop("norm.bias", None)
        incompatible = self.load_state_dict(state_dict, strict=False)
        return incompatible.missing_keys, incompatible.unexpected_keys


def convnext_base_aeanet(num_classes, **kwargs):
    return AEANet(
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        **kwargs
    )
