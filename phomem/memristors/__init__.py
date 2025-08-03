"""
Memristive device models with realistic physics and aging effects.
"""

from .devices import (
    PCMDevice,
    RRAMDevice, 
    PCMCrossbar,
    RRAMCrossbar
)

from .models import (
    DriftModel,
    ConductanceModel,
    SwitchingModel,
    AgingModel
)

from .spice import (
    SPICEInterface,
    generate_spice_netlist
)

__all__ = [
    "PCMDevice",
    "RRAMDevice",
    "PCMCrossbar", 
    "RRAMCrossbar",
    "DriftModel",
    "ConductanceModel",
    "SwitchingModel",
    "AgingModel",
    "SPICEInterface",
    "generate_spice_netlist"
]