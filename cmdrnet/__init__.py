"""CMDR-Net network modules."""

from .losses import LossWeights, compute_losses
from .model import (
    CMDRNet,
    CMDRNetConfig,
    HierarchicalMultimodalEncoder,
    PrototypeGuidedFusion,
    TemporalDepressionModeling,
    build_model,
)

__all__ = [
    "CMDRNet",
    "CMDRNetConfig",
    "HierarchicalMultimodalEncoder",
    "PrototypeGuidedFusion",
    "TemporalDepressionModeling",
    "build_model",
    "LossWeights",
    "compute_losses",
]

__version__ = "1.0.0"