import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleFPA(nn.Module):
    """Lightweight local/global aggregation for the deepest feature."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.local = ConvAct(in_channels, out_channels, 1)
        self.global_proj = ConvAct(in_channels, out_channels, 1)

    def forward(self, x):
        local = self.local(x)
        global_context = F.adaptive_avg_pool2d(x, 1)
        global_context = self.global_proj(global_context)
        return local + global_context


class SEWeight(nn.Module):
    """Return normalized channel weights rather than already-weighted features."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x):
        weight = self.pool(x)
        weight = F.relu(self.fc1(weight), inplace=True)
        return torch.sigmoid(self.fc2(weight))


class ContextGuideFusion(nn.Module):
    """Cross-complementary fusion guided by the concatenated global context."""

    def __init__(self, high_channels, current_channels):
        super().__init__()
        self.adjust = (
            nn.Identity()
            if high_channels == current_channels
            else nn.Sequential(
                nn.Conv2d(high_channels, current_channels, 1, bias=False),
                nn.BatchNorm2d(current_channels),
                nn.SiLU(inplace=True),
            )
        )
        self.se = SEWeight(current_channels * 2, reduction=16)

    def forward(self, high_feature, current_feature):
        high_feature = self.adjust(high_feature)
        merged = torch.cat([high_feature, current_feature], dim=1)
        high_weight, current_weight = torch.chunk(self.se(merged), 2, dim=1)
        enhanced_high = high_feature + current_feature * current_weight
        enhanced_current = current_feature + high_feature * high_weight
        return torch.cat([enhanced_high, enhanced_current], dim=1)


class ContextGuidedFeaturePyramid(nn.Module):
    """Top-down fusion of the middle and deepest ConvNeXt stages."""

    def __init__(self, p1_channels, p2_channels, p3_channels, out_channels):
        super().__init__()
        self.p3_enhance = SimpleFPA(p3_channels, out_channels)
        self.p3_refine = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.p2_lateral = nn.Conv2d(p2_channels, out_channels, 1)
        self.p2_fusion = ContextGuideFusion(out_channels, out_channels)
        self.p2_refine = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)

        self.p1_lateral = nn.Conv2d(p1_channels, out_channels, 1)
        self.p1_fusion = ContextGuideFusion(out_channels * 2, out_channels)
        self.p1_refine = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)

    def forward(self, p1, p2, p3):
        p3_base = self.p3_enhance(p3)
        p3_out = self.p3_refine(p3_base)

        p3_up = F.interpolate(p3_base, size=p2.shape[-2:], mode="bicubic", align_corners=False)
        p2_base = self.p2_lateral(p2)
        p2_fused = self.p2_fusion(p3_up, p2_base)
        p2_out = self.p2_refine(p2_fused)

        p2_up = F.interpolate(p2_fused, size=p1.shape[-2:], mode="bicubic", align_corners=False)
        p1_base = self.p1_lateral(p1)
        p1_fused = self.p1_fusion(p2_up, p1_base)
        p1_out = self.p1_refine(p1_fused)
        return p1_out, p2_out, p3_out


class LightweightChannelWeight(nn.Module):
    """SE-style channel attention with a 1/4 bottleneck."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.conv2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        weight = self.pool(x)
        weight = F.relu(self.conv1(weight), inplace=True)
        return torch.sigmoid(self.conv2(weight))


class BottomUpAttentionPropagation(nn.Module):
    """Propagate shallow detail attention toward the deepest semantic feature."""

    def __init__(self, channels):
        super().__init__()
        self.p1_attention = LightweightChannelWeight(channels)
        self.p2_attention = LightweightChannelWeight(channels)
        self.p3_attention = LightweightChannelWeight(channels)

    def forward(self, p1, p2, p3):
        a1 = self.p1_attention(p1)
        a2 = (self.p2_attention(p2) + a1) / 2.0
        a3 = (self.p3_attention(p3) + a2) / 2.0
        return p3 * a3, (a1, a2, a3)


class EfficientBilinearPooling(nn.Module):
    """Spatial compression followed by per-sample channel outer product."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, original_deep, enhanced_deep):
        if original_deep.size(0) != enhanced_deep.size(0):
            raise ValueError("EBP inputs must have the same batch size")
        if original_deep.shape[-2:] != enhanced_deep.shape[-2:]:
            enhanced_deep = F.interpolate(
                enhanced_deep,
                size=original_deep.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # EBP is deliberately computed in FP32. Under CUDA autocast the channel
        # outer product would otherwise run in FP16 and can overflow before the
        # subsequent L2 normalization has a chance to rescale it.
        device_type = original_deep.device.type
        autocast = (
            torch.amp.autocast(device_type, enabled=False)
            if device_type in ("cuda", "cpu")
            else contextlib.nullcontext()
        )
        with autocast:
            original_vector = F.adaptive_avg_pool2d(
                original_deep.float(), 1
            ).flatten(1)
            enhanced_vector = F.adaptive_avg_pool2d(
                enhanced_deep.float(), 1
            ).flatten(1)

            # normalize(a (x) b) == normalize(a) (x) normalize(b). Normalizing
            # the factors first bounds every multiplication and is more stable.
            original_vector = F.normalize(
                original_vector, p=2, dim=1, eps=self.eps
            )
            enhanced_vector = F.normalize(
                enhanced_vector, p=2, dim=1, eps=self.eps
            )
            return torch.bmm(
                original_vector.unsqueeze(2), enhanced_vector.unsqueeze(1)
            ).flatten(1)
