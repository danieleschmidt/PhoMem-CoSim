"""
PhoMem-CoSim: Photonic-Memristor Neuromorphic Co-Simulation Platform

A JAX-based differentiable simulator for hybrid photonic-memristive neural networks
with realistic device physics models and co-simulation capabilities.
"""

__version__ = "0.1.0"

from .neural import HybridNetwork
from .photonics import MachZehnderMesh, PhotoDetectorArray, ThermalPhaseShifter
from .memristors import PCMCrossbar, RRAMDevice, PCMDevice
from .simulator import MultiPhysicsSimulator, train, create_hardware_optimizer

__all__ = [
    "HybridNetwork",
    "MachZehnderMesh", 
    "PhotoDetectorArray",
    "ThermalPhaseShifter",
    "PCMCrossbar",
    "RRAMDevice", 
    "PCMDevice",
    "MultiPhysicsSimulator",
    "train",
    "create_hardware_optimizer"
]