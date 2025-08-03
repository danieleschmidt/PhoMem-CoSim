"""
Multi-physics simulator core for photonic-memristive co-simulation.
"""

from .core import (
    MultiPhysicsSimulator,
    OpticalSolver,
    ThermalSolver,
    ElectricalSolver
)

from .multiphysics import (
    ChipDesign,
    CoupledSimulation,
    FieldSolver
)

from .optimization import (
    NASOptimizer,
    HardwareOptimizer,
    ParetOptimizer
)

# Re-export training functions
from ..neural.training import (
    train,
    create_hardware_optimizer,
    hardware_aware_loss
)

__all__ = [
    "MultiPhysicsSimulator",
    "OpticalSolver",
    "ThermalSolver", 
    "ElectricalSolver",
    "ChipDesign",
    "CoupledSimulation",
    "FieldSolver",
    "NASOptimizer",
    "HardwareOptimizer",
    "ParetOptimizer",
    "train",
    "create_hardware_optimizer",
    "hardware_aware_loss"
]