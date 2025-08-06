"""
Input validation and data integrity utilities with comprehensive error handling.
"""

import numpy as np
import jax.numpy as jnp
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import warnings
import logging
import functools
import traceback
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, validation_context: Optional[Dict] = None):
        super().__init__(message)
        self.validation_context = validation_context or {}


class ValidationWarning(UserWarning):
    """Custom warning for validation issues."""
    pass


def validation_wrapper(func: Callable) -> Callable:
    """Decorator to add comprehensive error handling to validation functions."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError:
            raise  # Re-raise validation errors as-is
        except Exception as e:
            # Convert other exceptions to validation errors with context
            context = {
                'function': func.__name__,
                'args': str(args)[:200],  # Truncate for logging
                'kwargs': str(kwargs)[:200],
                'traceback': traceback.format_exc()
            }
            raise ValidationError(f"Validation failed in {func.__name__}: {e}", context)
    
    return wrapper


@contextmanager
def validation_context(context_name: str):
    """Context manager for validation operations."""
    logger.debug(f"Starting validation context: {context_name}")
    try:
        yield
        logger.debug(f"Completed validation context: {context_name}")
    except ValidationError as e:
        logger.error(f"Validation failed in context '{context_name}': {e}")
        e.validation_context['context'] = context_name
        raise
    except Exception as e:
        logger.error(f"Unexpected error in validation context '{context_name}': {e}")
        raise ValidationError(f"Unexpected error in {context_name}: {e}", {'context': context_name})


@validation_wrapper
def validate_input_array(
    array: Union[np.ndarray, jnp.ndarray],
    name: str,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_dtype: Optional[str] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allow_nan: bool = False,
    allow_inf: bool = False,
    allow_complex: bool = False
) -> Union[np.ndarray, jnp.ndarray]:
    """Validate input arrays with comprehensive checks and error handling."""
    
    with validation_context(f"validate_input_array({name})"):
        if array is None:
            raise ValidationError(f"{name} cannot be None")
        
        # Convert to array if needed
        if not isinstance(array, (np.ndarray, jnp.ndarray)):
            try:
                array = np.array(array)
            except Exception as e:
                raise ValidationError(
                    f"Cannot convert {name} to array: {e}",
                    {'input_type': type(array), 'input_value': str(array)[:100]}
                )
        
        # Check for empty arrays
        if array.size == 0:
            raise ValidationError(f"{name} is empty")
        
        # Check shape
        if expected_shape is not None:
            if array.shape != expected_shape:
                raise ValidationError(
                    f"{name} has shape {array.shape}, expected {expected_shape}",
                    {'actual_shape': array.shape, 'expected_shape': expected_shape}
                )
        
        # Check dtype and handle complex numbers
        if not allow_complex and np.iscomplexobj(array):
            raise ValidationError(f"{name} contains complex values but complex is not allowed")
        
        if expected_dtype is not None:
            if str(array.dtype) != expected_dtype:
                warnings.warn(
                    f"{name} has dtype {array.dtype}, expected {expected_dtype}. "
                    f"Converting automatically.",
                    ValidationWarning
                )
                try:
                    array = array.astype(expected_dtype)
                except Exception as e:
                    raise ValidationError(
                        f"Cannot convert {name} to {expected_dtype}: {e}",
                        {'current_dtype': str(array.dtype), 'target_dtype': expected_dtype}
                    )
        
        # Check for NaN values
        if not allow_nan:
            nan_count = np.sum(np.isnan(array))
            if nan_count > 0:
                raise ValidationError(
                    f"{name} contains {nan_count} NaN values",
                    {'nan_positions': np.where(np.isnan(array))[0][:10].tolist()}  # First 10 positions
                )
        
        # Check for infinite values
        if not allow_inf:
            inf_count = np.sum(np.isinf(array))
            if inf_count > 0:
                raise ValidationError(
                    f"{name} contains {inf_count} infinite values",
                    {'inf_positions': np.where(np.isinf(array))[0][:10].tolist()}
                )
        
        # Check value range
        if min_val is not None:
            below_min = np.sum(array < min_val)
            if below_min > 0:
                raise ValidationError(
                    f"{name} contains {below_min} values below minimum {min_val}",
                    {'min_value': float(np.min(array)), 'threshold': min_val}
                )
        
        if max_val is not None:
            above_max = np.sum(array > max_val)
            if above_max > 0:
                raise ValidationError(
                    f"{name} contains {above_max} values above maximum {max_val}",
                    {'max_value': float(np.max(array)), 'threshold': max_val}
                )
        
        return array


@validation_wrapper
def validate_device_parameters(params: Dict[str, Any], device_type: str) -> Dict[str, Any]:
    """Validate device-specific parameters with enhanced error handling."""
    
    with validation_context(f"validate_device_parameters({device_type})"):
        validated_params = {}
        
        if device_type == "photonic":
            # Validate photonic device parameters
            required_params = ["wavelength", "size"]
            optional_params = {
                "loss_db_cm": 0.5,
                "phase_shifter_type": "thermal",
                "power_per_pi": 20e-3,
                "response_time": 10e-6,
                "insertion_loss": 0.1,
                "crosstalk": 0.01
            }
            
            # Check required parameters
            missing_params = [p for p in required_params if p not in params]
            if missing_params:
                raise ValidationError(
                    f"Missing required parameters: {missing_params}",
                    {'required_params': required_params, 'provided_params': list(params.keys())}
                )
            
            for param in required_params:
                validated_params[param] = params[param]
            
            # Validate wavelength
            wavelength = params["wavelength"]
            if not isinstance(wavelength, (int, float)) or wavelength <= 0:
                raise ValidationError(f"Wavelength must be positive number, got {wavelength}")
            
            if not (800e-9 <= wavelength <= 2500e-9):
                warnings.warn(
                    f"Wavelength {wavelength*1e9:.0f}nm outside typical range (800-2500 nm)",
                    ValidationWarning
                )
            
            # Validate size
            size = params["size"]
            if not isinstance(size, int) or size <= 0:
                raise ValidationError(f"Size must be positive integer, got {size} (type: {type(size)})")
            
            if size > 1000:
                warnings.warn(f"Large photonic array size ({size}) may cause memory issues", ValidationWarning)
            
            # Validate phase shifter type
            valid_shifter_types = ["thermal", "plasma", "pcm", "mems"]
            phase_shifter_type = params.get("phase_shifter_type", "thermal")
            if phase_shifter_type not in valid_shifter_types:
                raise ValidationError(
                    f"Invalid phase shifter type: {phase_shifter_type}. Valid options: {valid_shifter_types}"
                )
            
            # Add optional parameters with validation
            for param, default in optional_params.items():
                value = params.get(param, default)
                
                # Parameter-specific validation
                if param == "loss_db_cm" and (not isinstance(value, (int, float)) or value < 0):
                    raise ValidationError(f"Loss must be non-negative number, got {value}")
                elif param == "power_per_pi" and (not isinstance(value, (int, float)) or value <= 0):
                    raise ValidationError(f"Power per pi must be positive, got {value}")
                elif param == "response_time" and (not isinstance(value, (int, float)) or value <= 0):
                    raise ValidationError(f"Response time must be positive, got {value}")
                elif param in ["insertion_loss", "crosstalk"] and (not isinstance(value, (int, float)) or not (0 <= value <= 1)):
                    raise ValidationError(f"{param} must be between 0 and 1, got {value}")
                
                validated_params[param] = value
        
        elif device_type == "memristive":
            # Validate memristive device parameters
            required_params = ["rows", "cols", "device_model"]
            optional_params = {
                "temperature": 300.0,
                "material": "GST225",
                "switching_energy": 100e-12,
                "retention_time": 1e6,
                "endurance_cycles": 1e8,
                "variability_sigma": 0.1
            }
            
            # Check required parameters
            missing_params = [p for p in required_params if p not in params]
            if missing_params:
                raise ValidationError(
                    f"Missing required parameters: {missing_params}",
                    {'required_params': required_params, 'provided_params': list(params.keys())}
                )
            
            for param in required_params:
                validated_params[param] = params[param]
            
            # Validate dimensions
            rows, cols = params["rows"], params["cols"]
            
            if not isinstance(rows, int) or not isinstance(cols, int):
                raise ValidationError(
                    f"Rows and cols must be integers, got rows={rows} (type: {type(rows)}), cols={cols} (type: {type(cols)})"
                )
            
            if rows <= 0 or cols <= 0:
                raise ValidationError(f"Rows and cols must be positive, got rows={rows}, cols={cols}")
            
            if rows > 10000 or cols > 10000:
                warnings.warn(
                    f"Large crossbar arrays ({rows}x{cols}) may require significant memory and time",
                    ValidationWarning
                )
            
            # Memory estimation
            estimated_memory_gb = (rows * cols * 8 * 4) / (1024**3)  # 4 arrays of float64
            if estimated_memory_gb > 1.0:
                warnings.warn(
                    f"Estimated memory usage: {estimated_memory_gb:.2f} GB",
                    ValidationWarning
                )
            
            # Validate device model
            valid_models = ["pcm_mushroom", "rram_hfo2", "rram_taox", "pcm_line", "ideal"]
            device_model = params["device_model"]
            if device_model not in valid_models:
                raise ValidationError(
                    f"Unknown device model: {device_model}. Valid options: {valid_models}"
                )
            
            # Add optional parameters with validation
            for param, default in optional_params.items():
                value = params.get(param, default)
                
                # Parameter-specific validation
                if param == "temperature":
                    if not isinstance(value, (int, float)) or not (0 < value < 1000):
                        raise ValidationError(f"Temperature must be between 0 and 1000 K, got {value}")
                elif param == "switching_energy":
                    if not isinstance(value, (int, float)) or value <= 0:
                        raise ValidationError(f"Switching energy must be positive, got {value}")
                elif param in ["retention_time", "endurance_cycles"]:
                    if not isinstance(value, (int, float)) or value <= 0:
                        raise ValidationError(f"{param} must be positive, got {value}")
                elif param == "variability_sigma":
                    if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                        raise ValidationError(f"Variability sigma must be between 0 and 1, got {value}")
                
                validated_params[param] = value
        
        else:
            raise ValidationError(
                f"Unknown device type: {device_type}",
                {'supported_types': ['photonic', 'memristive']}
            )
        
        return validated_params


@validation_wrapper
def validate_network_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate neural network configuration with comprehensive error handling."""
    
    with validation_context("validate_network_config"):
        validated_config = {}
        
        # Required parameters
        required_params = ["input_size", "hidden_sizes", "output_size"]
        missing_params = [p for p in required_params if p not in config]
        if missing_params:
            raise ValidationError(
                f"Missing required parameters: {missing_params}",
                {'required_params': required_params, 'provided_params': list(config.keys())}
            )
        
        for param in required_params:
            validated_config[param] = config[param]
        
        # Validate sizes
        input_size = config["input_size"]
        output_size = config["output_size"]
        hidden_sizes = config["hidden_sizes"]
        
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValidationError(
                f"input_size must be positive integer, got {input_size} (type: {type(input_size)})"
            )
        
        if not isinstance(output_size, int) or output_size <= 0:
            raise ValidationError(
                f"output_size must be positive integer, got {output_size} (type: {type(output_size)})"
            )
        
        if not isinstance(hidden_sizes, list) or len(hidden_sizes) == 0:
            raise ValidationError(
                f"hidden_sizes must be non-empty list, got {hidden_sizes} (type: {type(hidden_sizes)})"
            )
        
        for i, size in enumerate(hidden_sizes):
            if not isinstance(size, int) or size <= 0:
                raise ValidationError(
                    f"hidden_sizes[{i}] must be positive integer, got {size} (type: {type(size)})"
                )
        
        # Check for reasonable network size
        total_params = input_size * hidden_sizes[0]
        for i in range(len(hidden_sizes) - 1):
            total_params += hidden_sizes[i] * hidden_sizes[i + 1]
        total_params += hidden_sizes[-1] * output_size
        
        if total_params > 1e6:
            warnings.warn(
                f"Large network with {total_params:.0f} parameters may be slow",
                ValidationWarning
            )
        
        # Optional parameters with defaults
        optional_params = {
            "photonic_layers": [0],
            "memristive_layers": list(range(1, len(hidden_sizes) + 1)),
            "nonlinearity": "relu",
            "use_bias": True,
            "dropout_rate": 0.0,
            "batch_norm": False
        }
        
        for param, default in optional_params.items():
            validated_config[param] = config.get(param, default)
        
        # Validate layer indices
        total_layers = len(hidden_sizes) + 1  # +1 for output layer
        
        photonic_layers = validated_config["photonic_layers"]
        memristive_layers = validated_config["memristive_layers"]
        
        if not isinstance(photonic_layers, list):
            raise ValidationError(f"photonic_layers must be list, got {type(photonic_layers)}")
        
        if not isinstance(memristive_layers, list):
            raise ValidationError(f"memristive_layers must be list, got {type(memristive_layers)}")
        
        for layer_idx in photonic_layers:
            if not isinstance(layer_idx, int) or not (0 <= layer_idx < total_layers):
                raise ValidationError(f"Invalid photonic layer index: {layer_idx} (valid range: 0-{total_layers-1})")
        
        for layer_idx in memristive_layers:
            if not isinstance(layer_idx, int) or not (0 <= layer_idx < total_layers):
                raise ValidationError(f"Invalid memristive layer index: {layer_idx} (valid range: 0-{total_layers-1})")
        
        # Check for overlapping layers
        photonic_set = set(photonic_layers)
        memristive_set = set(memristive_layers)
        overlap = photonic_set.intersection(memristive_set)
        
        if overlap:
            raise ValidationError(f"Layers cannot be both photonic and memristive: {overlap}")
        
        # Validate nonlinearity
        valid_nonlinearities = ["relu", "tanh", "sigmoid", "gelu", "swish", "leaky_relu"]
        nonlinearity = validated_config["nonlinearity"]
        if nonlinearity not in valid_nonlinearities:
            raise ValidationError(
                f"Invalid nonlinearity: {nonlinearity}. Valid options: {valid_nonlinearities}"
            )
        
        # Validate dropout rate
        dropout_rate = validated_config["dropout_rate"]
        if not isinstance(dropout_rate, (int, float)) or not (0 <= dropout_rate < 1):
            raise ValidationError(f"Dropout rate must be between 0 and 1, got {dropout_rate}")
        
        return validated_config


@validation_wrapper
def validate_training_data(
    inputs: Union[np.ndarray, jnp.ndarray],
    targets: Union[np.ndarray, jnp.ndarray],
    input_size: int,
    output_size: int
) -> Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray]]:
    """Validate training data consistency with enhanced error handling."""
    
    with validation_context("validate_training_data"):
        # Validate inputs
        inputs = validate_input_array(
            inputs,
            "inputs",
            min_val=-100.0,
            max_val=100.0,
            allow_nan=False
        )
        
        # Validate targets
        targets = validate_input_array(
            targets,
            "targets",
            allow_nan=False
        )
        
        # Check dimensionality
        if inputs.ndim not in [1, 2]:
            raise ValidationError(
                f"Inputs must be 1D or 2D array, got shape {inputs.shape}",
                {'actual_dims': inputs.ndim, 'expected_dims': [1, 2]}
            )
        
        if targets.ndim not in [1, 2]:
            raise ValidationError(
                f"Targets must be 1D or 2D array, got shape {targets.shape}",
                {'actual_dims': targets.ndim, 'expected_dims': [1, 2]}
            )
        
        # Handle 1D inputs
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        if targets.ndim == 1:
            targets = targets.reshape(1, -1)
        
        # Check sizes
        batch_size = inputs.shape[0]
        input_features = inputs.shape[1]
        target_features = targets.shape[1]
        
        if input_features != input_size:
            raise ValidationError(
                f"Input feature size {input_features} != expected {input_size}",
                {'actual_features': input_features, 'expected_features': input_size}
            )
        
        if targets.shape[0] != batch_size:
            raise ValidationError(
                f"Batch size mismatch: inputs {batch_size}, targets {targets.shape[0]}"
            )
        
        if target_features != output_size:
            raise ValidationError(
                f"Output size {target_features} != expected {output_size}",
                {'actual_features': target_features, 'expected_features': output_size}
            )
        
        # Check for reasonable batch size
        if batch_size < 1:
            raise ValidationError("Batch size must be at least 1")
        
        if batch_size > 10000:
            warnings.warn(f"Large batch size ({batch_size}) may cause memory issues", ValidationWarning)
        
        # Check data quality
        input_std = np.std(inputs)
        if input_std < 1e-6:
            warnings.warn("Input data has very low variance, may affect training", ValidationWarning)
        
        target_range = np.max(targets) - np.min(targets)
        if target_range < 1e-6:
            warnings.warn("Target data has very small range, may affect training", ValidationWarning)
        
        return inputs, targets


class ConfigValidator:
    """Comprehensive configuration validator."""
    
    def __init__(self):
        self.validation_rules = {}
        self.validation_history = []
    
    def add_validation_rule(self, config_key: str, validator_func: Callable):
        """Add custom validation rule for configuration key."""
        self.validation_rules[config_key] = validator_func
    
    @validation_wrapper
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete configuration with custom rules."""
        
        with validation_context("validate_config"):
            validated_config = {}
            validation_errors = []
            validation_warnings = []
            
            # Apply built-in validation rules
            try:
                if 'network' in config:
                    validated_config['network'] = validate_network_config(config['network'])
            except ValidationError as e:
                validation_errors.append(f"Network config: {e}")
            
            # Apply custom validation rules
            for key, validator in self.validation_rules.items():
                if key in config:
                    try:
                        validated_config[key] = validator(config[key])
                    except ValidationError as e:
                        validation_errors.append(f"{key}: {e}")
                    except Exception as e:
                        validation_errors.append(f"{key}: Validation error - {e}")
            
            # Copy unvalidated keys (with warning)
            for key, value in config.items():
                if key not in validated_config:
                    validated_config[key] = value
                    if key not in ['network']:  # Don't warn about known keys
                        validation_warnings.append(f"No validation rule for key: {key}")
            
            # Record validation result
            validation_result = {
                'status': 'valid' if not validation_errors else 'invalid',
                'errors': validation_errors,
                'warnings': validation_warnings,
                'timestamp': logger.name
            }
            
            self.validation_history.append(validation_result)
            
            if validation_errors:
                raise ValidationError(
                    f"Configuration validation failed: {validation_errors}",
                    validation_result
                )
            
            if validation_warnings:
                for warning in validation_warnings:
                    warnings.warn(warning, ValidationWarning)
            
            return validated_config


class DataValidator:
    """Comprehensive data validation class with enhanced error handling."""
    
    def __init__(self, strict_mode: bool = True, max_history: int = 1000):
        self.strict_mode = strict_mode
        self.validation_history = []
        self.max_history = max_history
        self.error_count = 0
        self.warning_count = 0


# Global validator instances
_global_validator = DataValidator(strict_mode=True)
_global_config_validator = ConfigValidator()


def validate_network_architecture(network) -> List[str]:
    """Validate network architecture."""
    warnings = []
    try:
        if hasattr(network, 'layers'):
            for i, layer in enumerate(network.layers):
                if hasattr(layer, 'validate'):
                    layer_warnings = layer.validate()
                    warnings.extend([f"Layer {i}: {w}" for w in layer_warnings])
    except Exception as e:
        warnings.append(f"Architecture validation error: {e}")
    
    return warnings


def check_device_parameters(params: Dict[str, Any]) -> List[str]:
    """Check device parameters for validity."""
    warnings = []
    try:
        if 'device_type' in params:
            validated = validate_device_parameters(params, params['device_type'])
            return []  # No warnings if validation passes
    except ValidationError as e:
        warnings.append(str(e))
    except Exception as e:
        warnings.append(f"Parameter validation error: {e}")
    
    return warnings


def verify_simulation_setup(setup: Dict[str, Any]) -> List[str]:
    """Verify simulation setup."""
    warnings = []
    try:
        validator = _global_config_validator
        validated_setup = validator.validate_config(setup)
        return []  # No warnings if validation passes
    except ValidationError as e:
        warnings.append(str(e))
    except Exception as e:
        warnings.append(f"Setup validation error: {e}")
    
    return warnings


def get_validator() -> DataValidator:
    """Get the global validator instance."""
    return _global_validator


def get_config_validator() -> ConfigValidator:
    """Get the global configuration validator instance."""
    return _global_config_validator


def set_validation_mode(strict: bool):
    """Set global validation mode."""
    global _global_validator
    _global_validator.strict_mode = strict
    logger.info(f"Validation mode set to {'strict' if strict else 'permissive'}")