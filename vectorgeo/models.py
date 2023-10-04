import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlockConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResBlockConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity

        # Add SE block
        out = SEBlock(out.shape[1])(out)

        return out


class ResBlockDense(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResBlockDense, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

    def forward(self, x):
        identity = x
        out = F.relu(self.fc1(x))
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity

        out = SEBlock(out.shape[1])(out)
        
        return F.relu(out)


class ResnetTripletEmbedding(nn.Module):
    def __init__(
        self, input_shape, K, z_dim, num_filters, n_linear, num_dense_blocks=2, normalize=True
    ):
        super(ResnetTripletEmbedding, self).__init__()
        C, H, W = input_shape
        self.encoder = nn.Sequential()
        self.normalize = normalize


        n_downsamples = 0
        while H > 4:
            H, W = H // 2, W // 2
            n_downsamples += 1

        for _ in range(n_downsamples):
            self.encoder.add_module(
                f"conv_{_}", nn.Conv2d(C, num_filters, 3, padding=1)
            )
            for k in range(K):
                self.encoder.add_module(
                    f"resblock_conv_{k}", ResBlockConv(num_filters, num_filters, 3)
                )
            self.encoder.add_module(f"maxpool_{_}", nn.MaxPool2d(2, 2))
            C = num_filters

        self.encoder.add_module("flatten", nn.Flatten())
        self.encoder.add_module("dense", nn.Linear(num_filters * H * W, n_linear))

        self.dense_blocks = nn.Sequential(
            *[ResBlockDense(n_linear, n_linear) for _ in range(num_dense_blocks)]
        )

        self.output_layer = nn.Linear(n_linear, z_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.dense_blocks(x)

        if self.normalize:
            x = F.normalize(self.output_layer(x), dim=1)
        return x


def triplet_loss(y_pred, alpha=1.0):
    total_length = y_pred.shape[1]
    anchor, positive, negative = (
        y_pred[:, : total_length // 3],
        y_pred[:, total_length // 3 : 2 * total_length // 3],
        y_pred[:, 2 * total_length // 3 :],
    )

    pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
    neg_dist = torch.sum((anchor - negative) ** 2, dim=1)

    basic_loss = F.relu(pos_dist - neg_dist + alpha)

    loss = torch.sum(basic_loss)

    return loss, pos_dist, neg_dist


def initialize_triplet(input_shape, n_conv_blocks, embed_dim, num_filters, n_linear):
    embedding_network = ResnetTripletEmbedding(
        input_shape, n_conv_blocks, embed_dim, num_filters, n_linear
    )
    optimizer = torch.optim.Adam(embedding_network.parameters())

    return embedding_network, optimizer


def early_stopping(loss_history, tolerance=0.05, patience=3):
    """
    Check if the loss has converged. Uses the rolling mean of the loss over the last
    `patience` iterations to determine if the loss has converged. If the loss has
    converged, returns True. Otherwise, returns False.
    """
    if len(loss_history) < patience:
        return False
    
    if len(loss_history) > 2 * patience:
        loss_history = loss_history[-patience:]

    return np.mean(loss_history[-patience:]) - np.mean(loss_history[-2*patience:-patience]) < tolerance
