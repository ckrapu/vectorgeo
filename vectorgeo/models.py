import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return F.relu(out)

class ResnetTripletEmbedding(nn.Module):
    def __init__(self, input_shape, K, z_dim, num_filters, n_linear, num_dense_blocks=2):
        super(ResnetTripletEmbedding, self).__init__()
        C, H, W = input_shape
        self.encoder = nn.Sequential()

        n_downsamples = 0
        while H > 4:
            H, W = H // 2, W // 2
            n_downsamples += 1

        for _ in range(n_downsamples):
            self.encoder.add_module(f'conv_{_}', nn.Conv2d(C, num_filters, 3, padding=1))
            for k in range(K):
                self.encoder.add_module(f'resblock_conv_{k}', ResBlockConv(num_filters, num_filters, 3))
            self.encoder.add_module(f'maxpool_{_}', nn.MaxPool2d(2, 2))
            C = num_filters

        self.encoder.add_module('flatten', nn.Flatten())
        self.encoder.add_module('dense', nn.Linear(num_filters * H * W, n_linear))

        self.dense_blocks = nn.Sequential(
            *[ResBlockDense(n_linear, n_linear) for _ in range(num_dense_blocks)]
        )

        self.output_layer = nn.Linear(n_linear, z_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.dense_blocks(x)
        return self.output_layer(x)

def triplet_loss(y_pred, alpha=0.4, eta=0.1):
    total_length = y_pred.shape[1]
    anchor, positive, negative = y_pred[:, :total_length//3], y_pred[:, total_length//3:2*total_length//3], y_pred[:, 2*total_length//3:]

    pos_dist = torch.sum((anchor - positive)**2, dim=1)
    neg_dist = torch.sum((anchor - negative)**2, dim=1)

    l2_reg = torch.sum(anchor**2, dim=1) + torch.sum(positive**2, dim=1) + torch.sum(negative**2, dim=1)

    basic_loss = pos_dist - neg_dist + alpha
    loss = torch.sum(F.relu(basic_loss)) + eta * torch.sum(l2_reg)

    return loss

def initialize_triplet(input_shape, n_conv_blocks, embed_dim, num_filters, n_linear):
    embedding_network = ResnetTripletEmbedding(input_shape, n_conv_blocks, embed_dim, num_filters, n_linear)
    optimizer = torch.optim.Adam(embedding_network.parameters())

    return embedding_network, optimizer