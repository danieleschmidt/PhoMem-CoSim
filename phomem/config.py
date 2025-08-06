"""
Configuration management system for PhoMem-CoSim.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhotonicConfig:
    """Configuration for photonic components."""
    wavelength: float = 1550e-9
    loss_db_cm: float = 0.5
    phase_shifter_type: str = 'thermal'
    power_per_pi: float = 20e-3
    response_time: float = 10e-6


@dataclass  
class MemristiveConfig:
    """Configuration for memristive devices."""
    device_type: str = 'pcm_mushroom'
    material: str = 'GST225'
    switching_energy: float = 100e-12
    retention_time: float = 1e6  # seconds
    endurance_cycles: int = 1e8
    temperature: float = 300  # Kelvin


@dataclass
class NetworkConfig:
    """Configuration for hybrid network architecture."""
    input_size: int = 4
    hidden_sizes: list = field(default_factory=lambda: [16, 8])
    output_size: int = 2
    photonic_layers: list = field(default_factory=lambda: [0])  # indices
    memristive_layers: list = field(default_factory=lambda: [1, 2])  # indices
    nonlinearity: str = 'relu'


@dataclass
class SimulationConfig:
    """Configuration for multi-physics simulation."""
    optical_solver: str = 'BPM'
    thermal_solver: str = 'FEM'
    electrical_solver: str = 'SPICE'
    coupling: str = 'weak'
    duration: float = 1.0
    time_step: float = 1e-6
    save_fields: bool = False
    save_interval: float = 1e-3


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 10
    hardware_penalties: Dict[str, float] = field(default_factory=lambda: {
        'optical_loss': 0.1,
        'thermal_budget': 0.01,
        'aging_penalty': 0.001
    })


@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms."""
    optimizer: str = 'adam'
    grad_clip: float = 1.0
    weight_decay: float = 1e-4
    phase_shifter_bounds: tuple = field(default_factory=lambda: (-3.14159, 3.14159))
    memristor_bounds: tuple = field(default_factory=lambda: (1e3, 1e6))
    constraint_penalty: float = 1e3


@dataclass
class PhoMemConfig:
    """Master configuration class."""
    
    # Component configurations
    photonic: PhotonicConfig = field(default_factory=PhotonicConfig)
    memristive: MemristiveConfig = field(default_factory=MemristiveConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    # Global settings
    device: str = 'cpu'  # 'cpu', 'gpu', 'tpu'
    precision: str = 'float32'  # 'float16', 'float32', 'float64'
    seed: int = 42
    debug: bool = False
    
    def save(self, path: Union[str, Path]):
        """Save configuration to file."""
        path = Path(path)
        config_dict = asdict(self)
        
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'PhoMemConfig':
        """Load configuration from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            if path.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif path.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Handle nested dictionaries
        config = cls()
        for section, values in config_dict.items():
            if hasattr(config, section) and isinstance(values, dict):
                section_obj = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                    else:
                        logger.warning(f"Unknown config key: {section}.{key}")
            elif hasattr(config, section):
                setattr(config, section, values)
            else:
                logger.warning(f"Unknown config section: {section}")
        
        logger.info(f"Configuration loaded from {path}")
        return config
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with dictionary of values."""
        for key, value in updates.items():
            if '.' in key:
                # Handle nested updates like 'training.learning_rate'
                section, attr = key.split('.', 1)
                if hasattr(self, section):
                    section_obj = getattr(self, section)
                    if hasattr(section_obj, attr):
                        setattr(section_obj, attr, value)
                    else:
                        logger.warning(f"Unknown config attribute: {key}")
                else:
                    logger.warning(f"Unknown config section: {section}")
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown config key: {key}")
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        errors = []
        
        # Validate photonic parameters
        if self.photonic.wavelength <= 0:
            errors.append("Wavelength must be positive")
        
        if self.photonic.power_per_pi <= 0:
            errors.append("Power per pi must be positive")
        
        # Validate memristive parameters  
        if self.memristive.switching_energy <= 0:
            errors.append("Switching energy must be positive")
        
        if self.memristive.temperature <= 0:
            errors.append("Temperature must be positive")
        
        # Validate network architecture
        if self.network.input_size <= 0:
            errors.append("Input size must be positive")
        
        if self.network.output_size <= 0:
            errors.append("Output size must be positive")
        
        if not self.network.hidden_sizes:
            errors.append("Hidden sizes cannot be empty")
        
        # Validate training parameters
        if self.training.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        if self.training.epochs <= 0:
            errors.append("Epochs must be positive")
        
        if not (0 < self.training.validation_split < 1):
            errors.append("Validation split must be between 0 and 1")
        
        # Validate simulation parameters
        if self.simulation.duration <= 0:
            errors.append("Simulation duration must be positive")
        
        if self.simulation.time_step <= 0:
            errors.append("Time step must be positive")
        
        # Log errors
        if errors:
            for error in errors:
                logger.error(f"Config validation error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def get_device_config(self) -> Dict[str, Any]:
        """Get device-specific configuration."""
        return {
            'photonic': asdict(self.photonic),
            'memristive': asdict(self.memristive)
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration."""
        return {
            **asdict(self.training),
            **asdict(self.optimization)
        }


class ConfigManager:
    """Configuration management utilities."""
    
    @staticmethod
    def create_default_config(output_path: Union[str, Path]):
        """Create a default configuration file."""
        config = PhoMemConfig()
        config.save(output_path)
        return config
    
    @staticmethod
    def create_research_config(
        output_path: Union[str, Path],
        research_type: str = "general"
    ) -> PhoMemConfig:
        """Create configuration optimized for research applications."""
        config = PhoMemConfig()
        
        if research_type == "neuromorphic":
            # Optimize for neuromorphic computing
            config.network.hidden_sizes = [64, 32, 16]
            config.training.learning_rate = 5e-4
            config.training.epochs = 200
            config.memristive.device_type = 'rram_hfo2'
            config.simulation.save_fields = True
            
        elif research_type == "photonic":
            # Optimize for photonic neural networks
            config.photonic.phase_shifter_type = 'plasma'
            config.photonic.power_per_pi = 5e-3
            config.network.photonic_layers = [0, 1]
            config.network.memristive_layers = [2]
            
        elif research_type == "variability":
            # Optimize for variability studies
            config.simulation.save_fields = True
            config.training.hardware_penalties['aging_penalty'] = 0.01
            config.optimization.constraint_penalty = 100
            
        elif research_type == "scaling":
            # Optimize for scalability studies
            config.network.hidden_sizes = [128, 64, 32]
            config.training.batch_size = 64
            config.simulation.coupling = 'strong'
            config.device = 'gpu'
        
        config.save(output_path)
        return config
    
    @staticmethod
    def merge_configs(
        base_config: PhoMemConfig,
        override_config: Union[PhoMemConfig, Dict[str, Any]]
    ) -> PhoMemConfig:
        """Merge two configurations, with override taking precedence."""
        if isinstance(override_config, dict):
            base_config.update(override_config)
            return base_config
        
        # Merge PhoMemConfig objects
        base_dict = asdict(base_config)
        override_dict = asdict(override_config)
        
        # Deep merge dictionaries
        def deep_merge(base: Dict, override: Dict) -> Dict:
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        merged_dict = deep_merge(base_dict, override_dict)
        
        # Convert back to PhoMemConfig
        merged_config = PhoMemConfig()
        merged_config.update(merged_dict)
        
        return merged_config


# Global configuration instance
_global_config: Optional[PhoMemConfig] = None


def get_config() -> PhoMemConfig:
    """Get the global configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = PhoMemConfig()
    return _global_config


def set_config(config: PhoMemConfig):
    """Set the global configuration instance."""
    global _global_config
    _global_config = config
    logger.info("Global configuration updated")


def reset_config():
    """Reset to default configuration."""
    global _global_config
    _global_config = PhoMemConfig()
    logger.info("Configuration reset to defaults")