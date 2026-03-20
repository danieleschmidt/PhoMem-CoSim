"""
phomem_cosim — Photonic-Memristive Co-Simulator

Modules
-------
optical_layer         : OpticalLayer (MZI mesh unitary)
memristive_weight     : MemristiveWeight (PCM state machine)
co_differentiable_sim : CoDifferentiableSim (combined forward + gradient)
energy_model          : EnergyModel (optical + memristive energy accounting)
"""

from .optical_layer import OpticalLayer
from .memristive_weight import MemristiveWeight
from .co_differentiable_sim import CoDifferentiableSim
from .energy_model import EnergyModel

__all__ = [
    "OpticalLayer",
    "MemristiveWeight",
    "CoDifferentiableSim",
    "EnergyModel",
]

__version__ = "0.1.0"
