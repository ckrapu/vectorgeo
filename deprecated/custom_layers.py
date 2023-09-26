import torch
import torch.nn as nn
import torch.nn.functional as F

import keras.layers as layers


def conv_res_block(x, *conv_args, **conv_kwargs):
    u = x
    x = layers.Conv2D(*conv_args, **conv_kwargs)(u)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(*conv_args, **conv_kwargs)(u)
    x = layers.BatchNormalization()(x)
    x = x + u
    x = layers.ReLU()(x)
    return x


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, use_bn=True):
        super(ResidualBlock, self).__init__()
        self.use_bn = use_bn
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)

        if use_bn:
            self.bn1 = nn.BatchNorm1d(out_features)
            self.bn2 = nn.BatchNorm1d(out_features)

        # For the skip connection, if in and out features are not the same size.
        self.shortcut = (
            nn.Linear(in_features, out_features)
            if in_features != out_features
            else None
        )

    def forward(self, x):
        identity = x
        out = F.relu(self.linear1(x))
        if self.use_bn:
            out = self.bn1(out)
        out = self.linear2(out)
        if self.use_bn:
            out = self.bn2(out)

        # If the shortcut exists (in and out features differ)
        if self.shortcut:
            identity = self.shortcut(x)

        out += identity
        out = F.relu(out)
        return out


class ConvResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True
    ):
        super(ConvResidualBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

        # For the skip connection
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        if self.use_bn:
            out = self.bn1(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)

        # If the shortcut exists
        if self.shortcut:
            identity = self.shortcut(x)

        out += identity
        out = F.relu(out)
        return out
