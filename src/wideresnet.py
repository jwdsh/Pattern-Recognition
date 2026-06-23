import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBasicBlock(nn.Module):
    """WideResNet 的基本残差块：BN-ReLU-Conv-BN-ReLU-Conv + shortcut。"""

    def __init__(self, in_channels, out_channels, stride, dropout_rate=0.0):
        super().__init__()
        self.equal_shape = in_channels == out_channels and stride == 1

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.dropout_rate = dropout_rate

        # 当通道数或特征图尺寸变化时，用 1x1 卷积把 shortcut 调整到相同形状。
        if self.equal_shape:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )

    def forward(self, x):
        # WideResNet 使用 pre-activation 形式：先 BN/ReLU，再卷积。
        out = F.relu(self.bn1(x), inplace=True)
        shortcut = self.shortcut(out if not self.equal_shape else x)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(out)
        return out + shortcut


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=2, dropout_rate=0.0, num_classes=10):
        super().__init__()
        # WideResNet 深度满足 depth = 6n + 4；WRN-28-2 中 n = 4。
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth should satisfy depth = 6n + 4.")

        blocks_per_group = (depth - 4) // 6
        # widen_factor=2 表示三个残差组通道数变为 32、64、128。
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(
            3,
            channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        # CIFAR-10 输入是 32x32。三个残差组中后两组 stride=2，逐步降低空间分辨率。
        self.block1 = self._make_group(
            blocks_per_group,
            channels[0],
            channels[1],
            stride=1,
            dropout_rate=dropout_rate,
        )
        self.block2 = self._make_group(
            blocks_per_group,
            channels[1],
            channels[2],
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.block3 = self._make_group(
            blocks_per_group,
            channels[2],
            channels[3],
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.bn = nn.BatchNorm2d(channels[3])
        self.fc = nn.Linear(channels[3], num_classes)

        self._init_weights()

    def _make_group(self, num_blocks, in_channels, out_channels, stride, dropout_rate):
        # 每个 group 的第一个 block 可能负责下采样，其余 block 保持尺寸不变。
        layers = []
        for block_id in range(num_blocks):
            block_stride = stride if block_id == 0 else 1
            block_in_channels = in_channels if block_id == 0 else out_channels
            layers.append(
                WideBasicBlock(
                    block_in_channels,
                    out_channels,
                    stride=block_stride,
                    dropout_rate=dropout_rate,
                )
            )
        return nn.Sequential(*layers)

    def _init_weights(self):
        # 常见 CNN 初始化：卷积用 Kaiming，分类层用 Xavier，BN 初始为恒等缩放。
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out), inplace=True)
        # 自适应全局平均池化把每个通道压成一个数，之后接线性分类器。
        out = F.adaptive_avg_pool2d(out, output_size=1)
        out = torch.flatten(out, 1)
        return self.fc(out)
