"""
Hybrid neural network architectures combining photonic and memristive components.
"""

from .networks import (
    HybridNetwork,
    PhotonicLayer,
    MemristiveLayer,
    TransimpedanceAmplifier
)

from .architectures import (
    PhotonicMemristiveTransformer,
    PhotonicSNN,
    PhotonicLIFNeurons,
    MemristiveSTDPSynapses
)

from .training import (
    hardware_aware_loss,
    create_hardware_optimizer,
    train,
    train_stdp
)

__all__ = [
    "HybridNetwork",
    "PhotonicLayer",
    "MemristiveLayer", 
    "TransimpedanceAmplifier",
    "PhotonicMemristiveTransformer",
    "PhotonicSNN",
    "PhotonicLIFNeurons",
    "MemristiveSTDPSynapses",
    "hardware_aware_loss",
    "create_hardware_optimizer",
    "train",
    "train_stdp"
]