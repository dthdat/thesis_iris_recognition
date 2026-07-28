from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def maybe_checkpoint(module: nn.Module, x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and torch.is_grad_enabled() and x.requires_grad:
        return torch_checkpoint(module, x, use_reentrant=False)
    return module(x)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IrisIResNet50(nn.Module):
    """IResNet50 for 1-channel polar iris strips, optionally with MSFF."""

    fc_scale = 4 * 8

    def __init__(self, num_features: int = 512, dropout: float = 0.35, use_msff: bool = False):
        super().__init__()
        self.use_msff = bool(use_msff)
        self.inplanes = 64
        block = IBasicBlock
        layers = [3, 4, 14, 3]

        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-5)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-5)
        self.pool = nn.AdaptiveAvgPool2d((4, 8))

        if self.use_msff:
            self.fusion_conv = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
            self.fusion_bn = nn.BatchNorm2d(256, eps=1e-5)
            self.fusion_prelu = nn.PReLU(256)
            fc_channels = 768
        else:
            self.fusion_conv = None
            self.fusion_bn = None
            self.fusion_prelu = None
            fc_channels = 512

        self.dropout = nn.Dropout(p=float(dropout))
        self.fc = nn.Linear(fc_channels * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        self._initialize()

    def _make_layer(self, block: type[IBasicBlock], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-5),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, 0, 0.1)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        l3_feat = self.layer3(x)
        l4_feat = self.layer4(l3_feat)
        x = self.bn2(l4_feat)

        if self.use_msff:
            assert self.fusion_conv is not None and self.fusion_bn is not None and self.fusion_prelu is not None
            l3_down = self.fusion_conv(l3_feat)
            l3_down = self.fusion_bn(l3_down)
            l3_down = self.fusion_prelu(l3_down)
            x = torch.cat((l3_down, x), dim=1)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        embeds = F.normalize(x, p=2, dim=1)
        return embeds, l3_feat, l4_feat

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        embeds, _, _ = self.forward(x)
        return embeds


class IrisMobileNetV2(nn.Module):
    """MobileNetV2 embedding model adapted to 1-channel polar iris strips."""

    def __init__(self, num_features: int = 512, dropout: float = 0.35):
        super().__init__()
        net = tv_models.mobilenet_v2(weights=None)
        first = net.features[0][0]
        net.features[0][0] = nn.Conv2d(
            1,
            first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=False,
        )
        self.features_net = net.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=float(dropout))
        self.fc = nn.Linear(net.last_channel, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.features_net(x)
        pooled = self.pool(feat)
        x = torch.flatten(pooled, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        embeds = F.normalize(x, p=2, dim=1)
        return embeds, feat, feat

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        embeds, _, _ = self.forward(x)
        return embeds


class ConvBNPReLU(nn.Sequential):
    """MobileFaceNet convolution followed by batch norm and PReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, eps=1e-5),
            nn.PReLU(out_channels),
        )


class LinearConvBN(nn.Sequential):
    """MobileFaceNet linear convolution: convolution and batch norm only."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, eps=1e-5),
        )


class MobileFaceBottleneck(nn.Module):
    """Inverted residual bottleneck from the MobileFaceNet architecture."""

    def __init__(self, in_channels: int, out_channels: int, expansion: int, stride: int):
        super().__init__()
        hidden_channels = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        self.expand = ConvBNPReLU(in_channels, hidden_channels, kernel_size=1)
        self.depthwise = ConvBNPReLU(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=hidden_channels,
        )
        self.project = LinearConvBN(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.project(self.depthwise(self.expand(x)))
        return x + out if self.use_residual else out


class IrisMobileFaceNet(nn.Module):
    """MobileFaceNet adapted to notebook-compatible 1x64x512 iris strips.

    The stage layout follows the primary MobileFaceNet architecture. Its global
    depthwise convolution uses the final 4x32 polar feature map rather than
    replacing the learned spatial weighting with global average pooling.
    """

    output_stride = 16

    def __init__(
        self,
        num_features: int = 512,
        dropout: float = 0.35,
        input_size: tuple[int, int] = (64, 512),
    ):
        super().__init__()
        if any(size <= 0 or size % self.output_stride for size in input_size):
            raise ValueError(f"MobileFaceNet input_size must be positive and divisible by 16, got {input_size}")

        self.input_size = tuple(int(size) for size in input_size)
        self.gdc_size = tuple(size // self.output_stride for size in self.input_size)

        self.stem = ConvBNPReLU(1, 64, kernel_size=3, stride=2, padding=1)
        self.stem_depthwise = ConvBNPReLU(64, 64, kernel_size=3, padding=1, groups=64)
        self.stage1 = self._make_stage(64, 64, expansion=2, blocks=5, stride=2)
        self.stage2 = self._make_stage(64, 128, expansion=4, blocks=1, stride=2)
        self.stage3 = self._make_stage(128, 128, expansion=2, blocks=6, stride=1)
        self.stage4 = self._make_stage(128, 128, expansion=4, blocks=1, stride=2)
        self.stage5 = self._make_stage(128, 128, expansion=2, blocks=2, stride=1)
        self.conv_sep = ConvBNPReLU(128, 512, kernel_size=1)
        self.gdc = LinearConvBN(512, 512, kernel_size=self.gdc_size, groups=512)
        self.dropout = nn.Dropout(p=float(dropout))
        self.projection = nn.Conv2d(512, num_features, kernel_size=1, bias=False)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        self._initialize()

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        expansion: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [MobileFaceBottleneck(in_channels, out_channels, expansion, stride)]
        layers.extend(
            MobileFaceBottleneck(out_channels, out_channels, expansion, 1)
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if module.weight is not None and module.weight.requires_grad:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stem_depthwise(x)
        x = self.stage1(x)
        x = self.stage2(x)
        mid_feat = self.stage3(x)
        x = self.stage4(mid_feat)
        deep_feat = self.stage5(x)
        x = self.conv_sep(deep_feat)
        x = self.gdc(x)
        x = self.dropout(x)
        x = self.projection(x)
        x = torch.flatten(x, 1)
        x = self.features(x)
        embeds = F.normalize(x, p=2, dim=1)
        return embeds, mid_feat, deep_feat

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        embeds, _, _ = self.forward(x)
        return embeds


def build_model(config: dict[str, Any]) -> nn.Module:
    model_name = str(config.get("model_name", "iresnet50")).lower()
    embedding_dim = int(config.get("embedding_dim", 512))
    dropout = float(config.get("dropout", config.get("dropout_rate", 0.35)))
    if model_name in {"iresnet50", "iresnet", "arciris"}:
        return IrisIResNet50(
            num_features=embedding_dim,
            dropout=dropout,
            use_msff=bool(config.get("use_msff", False)),
        )
    if model_name in {"mobilenet_v2", "mobilenetv2", "mobilenet"}:
        return IrisMobileNetV2(num_features=embedding_dim, dropout=dropout)
    if model_name in {"mobilefacenet", "mobile_face_net", "mobileface"}:
        return IrisMobileFaceNet(
            num_features=embedding_dim,
            dropout=dropout,
            input_size=(int(config.get("polar_height", 64)), int(config.get("polar_width", 512))),
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def describe_model(model: nn.Module) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"parameters_total": total, "parameters_trainable": trainable}
