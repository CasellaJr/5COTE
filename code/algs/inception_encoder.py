"""
InceptionTime encoder building blocks (Fawaz et al., 2020), truncated
before the final classification layer -- used as a per-client embedding
extractor by BOTH fedinception_vertical.py (split learning) and
fedavg_zeropad_vertical.py (literal FedAvg). Factored out here so the
two don't maintain separate copies (same reasoning as federated_ridge.py
for the ridge-based families).

FusionHead lives here too: it's a plain 2-layer MLP over whatever
embedding_dim an encoder produces, so it's encoder-agnostic (used with
both InceptionTimeEncoder here and ResNet1DEncoder in
resnet1d_encoder.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=(9, 19, 39),
                 bottleneck_channels=32, use_bottleneck=True):
        super().__init__()
        self.use_bottleneck = use_bottleneck and in_channels > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            conv_in = bottleneck_channels
        else:
            self.bottleneck = None
            conv_in = in_channels

        self.convs = nn.ModuleList([
            nn.Conv1d(conv_in, n_filters, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.maxpool_conv = nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False)

        out_channels = n_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x_bottleneck = self.bottleneck(x) if self.use_bottleneck else x
        conv_outs = [conv(x_bottleneck) for conv in self.convs]
        pool_out = self.maxpool_conv(self.maxpool(x))
        out = torch.cat(conv_outs + [pool_out], dim=1)
        return F.relu(self.bn(out))


class ShortcutLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, residual, x):
        shortcut = self.bn(self.conv(residual))
        return F.relu(x + shortcut)


class InceptionTimeEncoder(nn.Module):
    """
    Truncated InceptionTime: `depth` Inception modules with a residual
    connection every 3 modules, followed by global average pooling.
    Outputs an embedding per series -- NOT class logits (the classifier
    lives in FusionHead, applied outside this encoder).
    """
    def __init__(self, in_channels, n_filters=32, depth=6,
                 kernel_sizes=(9, 19, 39), use_bottleneck=True, **kwargs):
        super().__init__()
        self.depth = depth
        module_out_channels = n_filters * (len(kernel_sizes) + 1)

        self.modules_list = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        channels = in_channels
        residual_input_channels = in_channels
        for d in range(depth):
            self.modules_list.append(
                InceptionModule(channels, n_filters=n_filters,
                                 kernel_sizes=kernel_sizes, use_bottleneck=use_bottleneck)
            )
            channels = module_out_channels
            if (d + 1) % 3 == 0:
                self.shortcuts.append(ShortcutLayer(residual_input_channels, channels))
                residual_input_channels = channels
            else:
                self.shortcuts.append(nn.Identity())  # placeholder, unused (see forward)

        self.embedding_dim = module_out_channels

    def forward(self, x):
        residual = x
        out = x
        for d in range(self.depth):
            out = self.modules_list[d](out)
            if (d + 1) % 3 == 0:
                out = self.shortcuts[d](residual, out)
                residual = out
        return out.mean(dim=-1)  # global average pooling over time -> (batch, embedding_dim)


class FusionHead(nn.Module):
    """
    Small MLP classifier head over an embedding (or concatenation of
    embeddings). Encoder-agnostic -- only needs in_dim to match whatever
    embedding_dim(s) feed into it. In fedinception_vertical.py this lives
    on the active party and is the only module that ever sees the labels;
    in fedavg_zeropad_vertical.py it's just the classifier half of the
    single shared model.
    """
    def __init__(self, in_dim, n_classes, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)