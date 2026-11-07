"""
Shared encoder registry: both fedinception_vertical.py (split learning)
and fedavg_zeropad_vertical.py (literal FedAvg) build their per-client
encoders through build_encoder() so a new encoder only needs to be added
here once to become available to both.

Both registered encoders share the same interface: constructor takes
`in_channels`, forward takes (batch, in_channels, length) and returns
(batch, embedding_dim), and the instance exposes `.embedding_dim`. Any
new encoder added to ENCODER_REGISTRY must follow the same contract.
"""

from algs.inception_encoder import InceptionTimeEncoder
from algs.resnet1d_encoder import ResNet1DEncoder

ENCODER_REGISTRY = {
    "inception": InceptionTimeEncoder,
    "resnet1d": ResNet1DEncoder,
}


def build_encoder(name, in_channels, n_filters=32, depth=6):
    """
    n_filters / depth only apply to "inception" -- ResNet1DEncoder's
    architecture (64->128->256->512 channels, 4 fixed stages) isn't
    parameterized the same way (see resnet1d_encoder.py), so they're
    silently ignored for "resnet1d".
    """
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"encoder must be one of {list(ENCODER_REGISTRY.keys())}, got {name!r}")
    encoder_cls = ENCODER_REGISTRY[name]
    if name == "inception":
        return encoder_cls(in_channels=in_channels, n_filters=n_filters, depth=depth)
    return encoder_cls(in_channels=in_channels)