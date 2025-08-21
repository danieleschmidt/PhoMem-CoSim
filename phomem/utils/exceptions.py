"""
Comprehensive exception hierarchy for PhoMem-CoSim with enhanced error handling and recovery.
"""

from typing import Dict, Any, Optional, List, Union
import traceback
from dataclasses import dataclass


@dataclass
class ErrorContext:
    """Context information for errors."""
    module: str
    function: str
    parameters: Dict[str, Any]
    timestamp: str
    stack_trace: str


class PhoMemError(Exception):
    """Base exception class for PhoMem-CoSim."""
    
    def __init__(self, 
                 message: str,
                 error_code: str = None,
                 context: Dict[str, Any] = None,
                 suggestions: List[str] = None):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.suggestions = suggestions or []
        self.stack_trace = traceback.format_exc()
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'context': self.context,
            'suggestions': self.suggestions,
            'stack_trace': self.stack_trace
        }
    
    def get_user_message(self) -> str:
        """Get user-friendly error message with suggestions."""
        msg = f"{self.message}"
        if self.suggestions:
            msg += "\n\nSuggestions:"
            for i, suggestion in enumerate(self.suggestions, 1):
                msg += f"\n  {i}. {suggestion}"
        return msg


# Device Physics Errors
class DevicePhysicsError(PhoMemError):
    """Errors related to device physics calculations."""
    pass


class MemristorError(DevicePhysicsError):
    """Errors specific to memristor operations."""
    
    def __init__(self, message: str, device_state: Dict[str, Any] = None, **kwargs):
        self.device_state = device_state or {}
        super().__init__(message, **kwargs)


class PhotonicError(DevicePhysicsError):
    """Errors specific to photonic components."""
    
    def __init__(self, message: str, wavelength: float = None, **kwargs):
        self.wavelength = wavelength
        super().__init__(message, **kwargs)


class PhaseShifterError(PhotonicError):
    """Errors related to phase shifter operations."""
    
    def __init__(self, message: str, phase_value: float = None, **kwargs):
        self.phase_value = phase_value
        context = kwargs.get('context', {})
        context['phase_value'] = phase_value
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class CrossbarError(MemristorError):
    """Errors related to crossbar array operations."""
    
    def __init__(self, message: str, array_size: tuple = None, **kwargs):
        self.array_size = array_size
        context = kwargs.get('context', {})
        context['array_size'] = array_size
        kwargs['context'] = context
        super().__init__(message, **kwargs)


# Simulation Errors
class SimulationError(PhoMemError):
    """Base class for simulation errors."""
    pass


class ConvergenceError(SimulationError):
    """Error when simulation fails to converge."""
    
    def __init__(self, message: str, 
                 max_iterations: int = None,
                 final_error: float = None,
                 target_error: float = None,
                 **kwargs):
        self.max_iterations = max_iterations
        self.final_error = final_error
        self.target_error = target_error
        
        context = kwargs.get('context', {})
        context.update({
            'max_iterations': max_iterations,
            'final_error': final_error,
            'target_error': target_error
        })
        kwargs['context'] = context
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Increase maximum iterations",
            "Reduce convergence tolerance",
            "Check initial conditions",
            "Verify boundary conditions",
            "Try different solver method"
        ])
        kwargs['suggestions'] = suggestions
        
        super().__init__(message, **kwargs)


class MultiPhysicsError(SimulationError):
    """Errors in multi-physics coupling."""
    
    def __init__(self, message: str, 
                 physics_domains: List[str] = None,
                 coupling_type: str = None,
                 **kwargs):
        self.physics_domains = physics_domains or []
        self.coupling_type = coupling_type
        
        context = kwargs.get('context', {})
        context.update({
            'physics_domains': physics_domains,
            'coupling_type': coupling_type
        })
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class SolverError(SimulationError):
    """Errors from physics solvers."""
    
    def __init__(self, message: str,
                 solver_type: str = None,
                 solver_method: str = None,
                 **kwargs):
        self.solver_type = solver_type
        self.solver_method = solver_method
        
        context = kwargs.get('context', {})
        context.update({
            'solver_type': solver_type,
            'solver_method': solver_method
        })
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


# Neural Network Errors
class NeuralNetworkError(PhoMemError):
    """Base class for neural network errors."""
    pass


class TrainingError(NeuralNetworkError):
    """Errors during neural network training."""
    
    def __init__(self, message: str,
                 epoch: int = None,
                 loss_value: float = None,
                 **kwargs):
        self.epoch = epoch
        self.loss_value = loss_value
        
        context = kwargs.get('context', {})
        context.update({
            'epoch': epoch,
            'loss_value': loss_value
        })
        kwargs['context'] = context
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Reduce learning rate",
            "Check for gradient explosion/vanishing",
            "Verify input data preprocessing",
            "Adjust network architecture",
            "Add regularization"
        ])
        kwargs['suggestions'] = suggestions
        
        super().__init__(message, **kwargs)


class ArchitectureError(NeuralNetworkError):
    """Errors in network architecture definition."""
    
    def __init__(self, message: str,
                 layer_info: Dict[str, Any] = None,
                 **kwargs):
        self.layer_info = layer_info or {}
        
        context = kwargs.get('context', {})
        context.update({
            'layer_info': layer_info
        })
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class HardwareConstraintError(NeuralNetworkError):
    """Errors when hardware constraints are violated."""
    
    def __init__(self, message: str,
                 constraint_type: str = None,
                 current_value: float = None,
                 limit_value: float = None,
                 **kwargs):
        self.constraint_type = constraint_type
        self.current_value = current_value
        self.limit_value = limit_value
        
        context = kwargs.get('context', {})
        context.update({
            'constraint_type': constraint_type,
            'current_value': current_value,
            'limit_value': limit_value
        })
        kwargs['context'] = context
        
        suggestions = kwargs.get('suggestions', [])
        if constraint_type == 'power':
            suggestions.extend([
                "Reduce network size",
                "Use lower power phase shifters",
                "Optimize phase settings",
                "Consider duty cycling"
            ])
        elif constraint_type == 'optical_loss':
            suggestions.extend([
                "Reduce number of components",
                "Use lower loss materials",
                "Optimize waveguide design",
                "Add optical amplification"
            ])
        kwargs['suggestions'] = suggestions
        
        super().__init__(message, **kwargs)


# Input/Output Errors
class InputValidationError(PhoMemError):
    """Errors in input parameter validation."""
    
    def __init__(self, message: str,
                 parameter_name: str = None,
                 parameter_value: Any = None,
                 expected_type: type = None,
                 expected_range: tuple = None,
                 **kwargs):
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.expected_type = expected_type
        self.expected_range = expected_range
        
        context = kwargs.get('context', {})
        context.update({
            'parameter_name': parameter_name,
            'parameter_value': parameter_value,
            'expected_type': expected_type.__name__ if expected_type else None,
            'expected_range': expected_range
        })
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class DimensionMismatchError(InputValidationError):
    """Errors when array dimensions don't match."""
    
    def __init__(self, message: str,
                 expected_shape: tuple = None,
                 actual_shape: tuple = None,
                 **kwargs):
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        
        context = kwargs.get('context', {})
        context.update({
            'expected_shape': expected_shape,
            'actual_shape': actual_shape
        })
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


# File I/O Errors
class FileError(PhoMemError):
    """Base class for file-related errors."""
    pass


class ConfigurationError(FileError):
    """Errors in configuration file handling."""
    
    def __init__(self, message: str,
                 config_file: str = None,
                 config_section: str = None,
                 **kwargs):
        self.config_file = config_file
        self.config_section = config_section
        
        context = kwargs.get('context', {})
        context.update({
            'config_file': config_file,
            'config_section': config_section
        })
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


class ExportError(FileError):
    """Errors during data export operations."""
    
    def __init__(self, message: str,
                 export_format: str = None,
                 file_path: str = None,
                 **kwargs):
        self.export_format = export_format
        self.file_path = file_path
        
        context = kwargs.get('context', {})
        context.update({
            'export_format': export_format,
            'file_path': file_path
        })
        kwargs['context'] = context
        
        super().__init__(message, **kwargs)


# Hardware Interface Errors
class HardwareInterfaceError(PhoMemError):
    """Errors in hardware interface operations."""
    pass


class SPICEError(HardwareInterfaceError):
    """Errors in SPICE simulation interface."""
    
    def __init__(self, message: str,
                 spice_command: str = None,
                 return_code: int = None,
                 **kwargs):
        self.spice_command = spice_command
        self.return_code = return_code
        
        context = kwargs.get('context', {})
        context.update({
            'spice_command': spice_command,
            'return_code': return_code
        })
        kwargs['context'] = context
        
        suggestions = kwargs.get('suggestions', [])
        suggestions.extend([
            "Check ngspice installation",
            "Verify netlist syntax",
            "Check file permissions",
            "Ensure temporary directory is writable"
        ])
        kwargs['suggestions'] = suggestions
        
        super().__init__(message, **kwargs)


# Utility Functions
def handle_jax_errors(func):
    """Decorator to handle common JAX errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ImportError as e:
            if 'jax' in str(e).lower():
                raise PhoMemError(
                    "JAX not installed or not properly configured",
                    error_code="JAX_MISSING",
                    context={'original_error': str(e)},
                    suggestions=[
                        "Install JAX: pip install jax jaxlib",
                        "For GPU support: pip install jax[cuda12_pip]",
                        "Check JAX installation guide"
                    ]
                )
            raise
        except Exception as e:
            # Re-raise as PhoMem error with context
            if isinstance(e, PhoMemError):
                raise
            raise PhoMemError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                context={'function': func.__name__, 'args': str(args)[:100]}
            )
    return wrapper


def validate_array_input(array, name: str, expected_shape: tuple = None, 
                        expected_dtype = None, allow_none: bool = False):
    """Validate array inputs with detailed error messages."""
    if array is None and allow_none:
        return
    
    if array is None:
        raise InputValidationError(
            f"Parameter '{name}' cannot be None",
            parameter_name=name,
            parameter_value=None
        )
    
    # Check if it's array-like
    if not hasattr(array, 'shape'):
        raise InputValidationError(
            f"Parameter '{name}' must be array-like (have .shape attribute)",
            parameter_name=name,
            parameter_value=type(array).__name__,
            expected_type=type(array)
        )
    
    # Check shape
    if expected_shape is not None and array.shape != expected_shape:
        raise DimensionMismatchError(
            f"Parameter '{name}' has incorrect shape",
            parameter_name=name,
            expected_shape=expected_shape,
            actual_shape=array.shape
        )
    
    # Check dtype
    if expected_dtype is not None and array.dtype != expected_dtype:
        raise InputValidationError(
            f"Parameter '{name}' has incorrect dtype",
            parameter_name=name,
            parameter_value=str(array.dtype),
            expected_type=expected_dtype
        )


def validate_range(value: Union[int, float], name: str, 
                  min_val: float = None, max_val: float = None,
                  inclusive: bool = True):
    """Validate numeric ranges with detailed error messages."""
    if min_val is not None:
        if inclusive and value < min_val:
            raise InputValidationError(
                f"Parameter '{name}' must be >= {min_val}",
                parameter_name=name,
                parameter_value=value,
                expected_range=(min_val, max_val)
            )
        elif not inclusive and value <= min_val:
            raise InputValidationError(
                f"Parameter '{name}' must be > {min_val}",
                parameter_name=name,
                parameter_value=value,
                expected_range=(min_val, max_val)
            )
    
    if max_val is not None:
        if inclusive and value > max_val:
            raise InputValidationError(
                f"Parameter '{name}' must be <= {max_val}",
                parameter_name=name,
                parameter_value=value,
                expected_range=(min_val, max_val)
            )
        elif not inclusive and value >= max_val:
            raise InputValidationError(
                f"Parameter '{name}' must be < {max_val}",
                parameter_name=name,
                parameter_value=value,
                expected_range=(min_val, max_val)
            )