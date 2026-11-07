"""
ResNet1D encoder, adapted from user-provided ResNet1DMultivariate code,
truncated before the final classification layer so it can be used as a
pluggable ENCODER for fedavg_zeropad_vertical.py -- alternative to
InceptionTimeEncoder (see fedinception_vertical.py), selectable via
--encoder resnet1d in vertical.py.

Architecture is a faithful port of the original (4 residual stages,
64->128->256->512 channels, stride-2 downsampling in every stage after
the first, kernel_size=7 stride=2 stem). Two changes from the original:
1. No final nn.Linear / num_classes -- that lives in FusionHead
   (shared with the InceptionTime path), and this exposes
   `embedding_dim` (=512) for interface consistency.
2. `init_weights` in the original checks `isinstance(m, nn.Conv2d)` /
   `nn.BatchNorm2d` -- but every layer in this network is Conv1d /
   BatchNorm1d, so those branches never matched and the intended
   Kaiming/constant initialization never actually applied (only the
   nn.Linear branch could ever fire). This is fixed here to check
   Conv1d/BatchNorm1d as actually used, so the intended initialization
   takes effect. Purely an initialization-statistics difference --  the
   architecture/computation graph is otherwise identical to the
   original.

CAVEAT: this architecture downsamples the time axis by a factor of
2^4 = 16 (stem stride 2, then 3 more stride-2 stages) before the final
global average pool. On the SHORT series typical of UEA/UCR TSC
benchmarks (some only a few dozen timesteps), this can shrink the
temporal dimension to near zero or trigger a shape error inside a deep
Conv1d layer -- it was originally designed for longer 1D signals (e.g.
ECG). If you hit a shape/size error on a short-series dataset, that's
almost certainly why; the fix is dataset-dependent (fewer stride-2
stages, or only try this encoder on longer-series datasets) and hasn't
been engineered here since it wasn't asked for.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # residual connection
        out = self.relu(out)
        return out


class ResNet1DEncoder(nn.Module):
    """
    Truncated ResNet1DMultivariate: everything up to (and including)
    global average pooling + flatten, no final nn.Linear. Outputs a
    512-dim embedding per series -- the classifier head (FusionHead)
    lives outside, shared with the InceptionTime path.
    """

    def __init__(self, in_channels=1, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        self.embedding_dim = 512
        self._init_weights()

    def _init_weights(self):
        # NOTE: checks Conv1d/BatchNorm1d (what this network actually
        # uses) -- see module docstring point 2.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride),
                  ResidualBlock(out_channels, out_channels)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool1d(x, 1)
        x = torch.flatten(x, 1)
        return x