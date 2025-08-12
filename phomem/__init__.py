"""
PhoMem-CoSim: Photonic-Memristor Neuromorphic Co-Simulation Platform

A comprehensive JAX-based differentiable simulator for hybrid photonic-memristive neural networks
with realistic device physics models, multi-physics co-simulation, and advanced optimization.
"""

__version__ = "0.1.0"

# Core neural network components
from .neural import HybridNetwork
from .neural.networks import PhotonicLayer, MemristiveLayer
# from .neural.training import HardwareAwareTrainer  # TODO: Implement HardwareAwareTrainer

# Photonic components
from .photonics import MachZehnderMesh, PhotoDetectorArray, ThermalPhaseShifter
from .photonics.components import MachZehnderInterferometer, PhaseShifter

# Memristive components  
from .memristors import PCMCrossbar, RRAMDevice, PCMDevice
from .memristors.models import PCMModel, RRAMModel

# Simulation engines
from .simulator import MultiPhysicsSimulator, train, create_hardware_optimizer
# from .simulator.core import SimulationEngine  # TODO: Check class name

# Configuration and management
from .config import PhoMemConfig, ConfigManager
from .batch import BatchProcessor, BatchJob
from .calibration import CalibrationManager, CalibrationData

# Optimization and research
from .optimization import MultiObjectiveOptimizer, NeuralArchitectureSearch
from .research import ResearchFramework, QuantumInspiredOptimizer
from .benchmarking import PerformanceBenchmark, run_comprehensive_benchmark

# Utilities
from .utils.validation import get_validator, ValidationError
from .utils.security import get_security_validator, SecurityError
from .utils.performance import PerformanceOptimizer, MemoryManager
from .utils.logging import setup_logging, get_logger

__all__ = [
    # Core components
    "HybridNetwork",
    "PhotonicLayer", 
    "MemristiveLayer",
    # "HardwareAwareTrainer",  # TODO: Implement
    
    # Photonic devices
    "MachZehnderMesh", 
    "PhotoDetectorArray",
    "ThermalPhaseShifter",
    "MachZehnderInterferometer",
    "PhaseShifter",
    
    # Memristive devices
    "PCMCrossbar",
    "RRAMDevice", 
    "PCMDevice",
    "PCMModel",
    "RRAMModel",
    
    # Simulation
    "MultiPhysicsSimulator",
    # "SimulationEngine",  # TODO: Check class name
    "train",
    "create_hardware_optimizer",
    
    # Configuration
    "PhoMemConfig",
    "ConfigManager",
    
    # Batch processing
    "BatchProcessor",
    "BatchJob",
    
    # Calibration
    "CalibrationManager",
    "CalibrationData",
    
    # Optimization
    "MultiObjectiveOptimizer",
    "NeuralArchitectureSearch",
    
    # Research
    "ResearchFramework",
    "QuantumInspiredOptimizer",
    
    # Benchmarking
    "PerformanceBenchmark",
    "run_comprehensive_benchmark",
    
    # Utilities
    "get_validator",
    "ValidationError",
    "get_security_validator", 
    "SecurityError",
    "PerformanceOptimizer",
    "MemoryManager",
    "setup_logging",
    "get_logger"
]