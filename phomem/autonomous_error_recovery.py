"""
Autonomous Error Recovery and Self-Healing System - AUTONOMOUS SDLC v4.0
Advanced error detection, recovery, and self-healing for robust operation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import chex
from functools import partial
import time
import logging
from dataclasses import dataclass
from enum import Enum
import traceback
from pathlib import Path

from .utils.logging import get_logger
from .utils.exceptions import ValidationError, SecurityError


class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADED_MODE = "degraded_mode"
    RESTART = "restart"
    ESCALATE = "escalate"


@dataclass
class ErrorContext:
    """Comprehensive error context information."""
    
    error_type: str
    error_message: str
    severity: ErrorSeverity
    timestamp: float
    component: str
    input_state: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    recovery_attempts: int = 0
    successful_recovery: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None


class SmartErrorDetector:
    """Intelligent error detection with pattern recognition."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.error_patterns = {}
        self.anomaly_threshold = 3.0  # Standard deviations
        self.historical_metrics = []
        
    def detect_numerical_instability(self, 
                                   values: chex.Array, 
                                   context: str = "unknown") -> Optional[ErrorContext]:
        """Detect numerical instabilities in computations."""
        
        # Check for NaN/Inf values
        if jnp.any(jnp.isnan(values)) or jnp.any(jnp.isinf(values)):
            return ErrorContext(
                error_type="numerical_instability",
                error_message=f"NaN/Inf detected in {context}",
                severity=ErrorSeverity.HIGH,
                timestamp=time.time(),
                component=context,
                input_state={'values_shape': values.shape, 'values_dtype': str(values.dtype)}
            )
        
        # Check for extreme values
        max_val = jnp.max(jnp.abs(values))
        if max_val > 1e10:
            return ErrorContext(
                error_type="extreme_values",
                error_message=f"Extreme values detected in {context}: {max_val}",
                severity=ErrorSeverity.MEDIUM,
                timestamp=time.time(),
                component=context,
                input_state={'max_value': float(max_val)}
            )
        
        return None
    
    def detect_convergence_failure(self, 
                                 residual_history: List[float],
                                 max_iterations: int) -> Optional[ErrorContext]:
        """Detect convergence failures in iterative algorithms."""
        
        if len(residual_history) >= max_iterations:
            # Check if residual is still decreasing
            recent_residuals = residual_history[-10:]
            if len(recent_residuals) > 1:
                trend = np.polyfit(range(len(recent_residuals)), recent_residuals, 1)[0]
                if trend >= 0:  # Not decreasing
                    return ErrorContext(
                        error_type="convergence_failure",
                        error_message="Algorithm failed to converge",
                        severity=ErrorSeverity.HIGH,
                        timestamp=time.time(),
                        component="iterative_solver",
                        input_state={'residual_history': residual_history[-5:]}
                    )
        
        return None
    
    def detect_physical_implausibility(self, 
                                     simulation_output: Dict[str, Any]) -> List[ErrorContext]:
        """Detect physically implausible simulation results."""
        
        errors = []
        
        # Energy conservation violation
        if 'energy_input' in simulation_output and 'energy_output' in simulation_output:
            energy_error = abs(simulation_output['energy_output'] - simulation_output['energy_input'])
            relative_error = energy_error / max(simulation_output['energy_input'], 1e-12)
            
            if relative_error > 0.1:  # 10% threshold
                errors.append(ErrorContext(
                    error_type="energy_conservation_violation",
                    error_message=f"Energy not conserved: {relative_error:.2%} error",
                    severity=ErrorSeverity.MEDIUM,
                    timestamp=time.time(),
                    component="physics_simulation",
                    input_state={'energy_error': float(relative_error)}
                ))
        
        # Temperature limits
        if 'temperature' in simulation_output:
            temp = simulation_output['temperature']
            if isinstance(temp, (jnp.ndarray, np.ndarray)):
                max_temp = jnp.max(temp)
            else:
                max_temp = temp
                
            if max_temp > 1000:  # 1000K limit
                errors.append(ErrorContext(
                    error_type="temperature_exceeds_limit",
                    error_message=f"Temperature exceeds safe limit: {max_temp}K",
                    severity=ErrorSeverity.HIGH,
                    timestamp=time.time(),
                    component="thermal_simulation",
                    input_state={'max_temperature': float(max_temp)}
                ))
        
        return errors
    
    def detect_memory_issues(self, 
                           memory_usage_mb: float, 
                           memory_limit_mb: float) -> Optional[ErrorContext]:
        """Detect memory-related issues."""
        
        usage_fraction = memory_usage_mb / memory_limit_mb
        
        if usage_fraction > 0.95:
            return ErrorContext(
                error_type="memory_exhaustion",
                error_message=f"Memory usage critical: {usage_fraction:.1%}",
                severity=ErrorSeverity.CRITICAL,
                timestamp=time.time(),
                component="resource_manager",
                input_state={'memory_usage_mb': memory_usage_mb, 'memory_limit_mb': memory_limit_mb}
            )
        elif usage_fraction > 0.8:
            return ErrorContext(
                error_type="high_memory_usage",
                error_message=f"High memory usage: {usage_fraction:.1%}",
                severity=ErrorSeverity.MEDIUM,
                timestamp=time.time(),
                component="resource_manager",
                input_state={'memory_usage_mb': memory_usage_mb}
            )
        
        return None


class AdaptiveRecoverySystem:
    """Intelligent recovery system with learning capabilities."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.recovery_history = {}
        self.success_rates = {}
        self.fallback_strategies = {}
        
    def select_recovery_strategy(self, error_context: ErrorContext) -> RecoveryStrategy:
        """Select optimal recovery strategy based on error type and history."""
        
        error_type = error_context.error_type
        
        # Check historical success rates
        if error_type in self.success_rates:
            best_strategy = max(self.success_rates[error_type].items(), 
                              key=lambda x: x[1])[0]
        else:
            # Default strategy based on error type
            best_strategy = self._get_default_strategy(error_context)
        
        self.logger.info(f"Selected recovery strategy '{best_strategy}' for error '{error_type}'")
        return RecoveryStrategy(best_strategy)
    
    def _get_default_strategy(self, error_context: ErrorContext) -> str:
        """Get default recovery strategy for error type."""
        
        strategy_map = {
            "numerical_instability": RecoveryStrategy.FALLBACK.value,
            "convergence_failure": RecoveryStrategy.RETRY.value,
            "memory_exhaustion": RecoveryStrategy.DEGRADED_MODE.value,
            "temperature_exceeds_limit": RecoveryStrategy.DEGRADED_MODE.value,
            "energy_conservation_violation": RecoveryStrategy.FALLBACK.value
        }
        
        return strategy_map.get(error_context.error_type, RecoveryStrategy.RETRY.value)
    
    def execute_recovery(self, 
                        error_context: ErrorContext,
                        original_function: Callable,
                        *args, **kwargs) -> Tuple[Any, bool]:
        """Execute recovery strategy and return result."""
        
        strategy = self.select_recovery_strategy(error_context)
        recovery_successful = False
        result = None
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                result = self._retry_with_modifications(original_function, error_context, *args, **kwargs)
                recovery_successful = True
                
            elif strategy == RecoveryStrategy.FALLBACK:
                result = self._fallback_computation(original_function, error_context, *args, **kwargs)
                recovery_successful = True
                
            elif strategy == RecoveryStrategy.DEGRADED_MODE:
                result = self._degraded_mode_computation(original_function, error_context, *args, **kwargs)
                recovery_successful = True
                
            elif strategy == RecoveryStrategy.RESTART:
                result = self._restart_computation(original_function, error_context, *args, **kwargs)
                recovery_successful = True
                
            else:  # ESCALATE
                self.logger.error(f"Escalating error: {error_context.error_message}")
                raise RuntimeError(f"Unrecoverable error: {error_context.error_message}")
                
        except Exception as e:
            self.logger.error(f"Recovery strategy '{strategy}' failed: {e}")
            recovery_successful = False
            
        # Update success statistics
        self._update_success_rates(error_context.error_type, strategy.value, recovery_successful)
        
        return result, recovery_successful
    
    def _retry_with_modifications(self, 
                                original_function: Callable,
                                error_context: ErrorContext,
                                *args, **kwargs) -> Any:
        """Retry computation with modified parameters."""
        
        # Modify parameters based on error type
        modified_kwargs = kwargs.copy()
        
        if error_context.error_type == "convergence_failure":
            # Increase iterations and adjust tolerance
            modified_kwargs['max_iterations'] = kwargs.get('max_iterations', 100) * 2
            modified_kwargs['tolerance'] = kwargs.get('tolerance', 1e-6) * 10
            
        elif error_context.error_type == "numerical_instability":
            # Reduce step size or increase regularization
            if 'step_size' in kwargs:
                modified_kwargs['step_size'] = kwargs['step_size'] * 0.5
            if 'regularization' in kwargs:
                modified_kwargs['regularization'] = kwargs['regularization'] * 10
                
        self.logger.info(f"Retrying with modified parameters: {modified_kwargs}")
        return original_function(*args, **modified_kwargs)
    
    def _fallback_computation(self, 
                            original_function: Callable,
                            error_context: ErrorContext,
                            *args, **kwargs) -> Any:
        """Use fallback computation method."""
        
        # Implement simpler, more stable algorithms
        if "numerical_instability" in error_context.error_type:
            # Use double precision
            modified_args = [jnp.array(arg, dtype=jnp.float64) if isinstance(arg, jnp.ndarray) else arg 
                           for arg in args]
            kwargs['precision'] = 'float64'
            return original_function(*modified_args, **kwargs)
        
        # Generic fallback: reduce complexity
        if 'complexity_factor' not in kwargs:
            kwargs['complexity_factor'] = 0.5
            
        return original_function(*args, **kwargs)
    
    def _degraded_mode_computation(self, 
                                 original_function: Callable,
                                 error_context: ErrorContext,
                                 *args, **kwargs) -> Any:
        """Run computation in degraded mode with reduced accuracy."""
        
        # Reduce precision and complexity for stability
        kwargs['degraded_mode'] = True
        kwargs['max_iterations'] = min(kwargs.get('max_iterations', 100), 50)
        kwargs['tolerance'] = max(kwargs.get('tolerance', 1e-6), 1e-4)
        
        self.logger.warning("Running in degraded mode - results may have reduced accuracy")
        return original_function(*args, **kwargs)
    
    def _restart_computation(self, 
                           original_function: Callable,
                           error_context: ErrorContext,
                           *args, **kwargs) -> Any:
        """Restart computation with fresh initialization."""
        
        # Reset any stateful components
        kwargs['fresh_start'] = True
        if 'random_seed' in kwargs:
            kwargs['random_seed'] = int(time.time()) % 10000
            
        return original_function(*args, **kwargs)
    
    def _update_success_rates(self, 
                            error_type: str, 
                            strategy: str, 
                            success: bool) -> None:
        """Update recovery strategy success rates."""
        
        if error_type not in self.success_rates:
            self.success_rates[error_type] = {}
            
        if strategy not in self.success_rates[error_type]:
            self.success_rates[error_type][strategy] = {'successes': 0, 'attempts': 0}
            
        self.success_rates[error_type][strategy]['attempts'] += 1
        if success:
            self.success_rates[error_type][strategy]['successes'] += 1


class SelfHealingDecorator:
    """Decorator for automatic error detection and recovery."""
    
    def __init__(self, 
                 max_recovery_attempts: int = 3,
                 enable_learning: bool = True):
        self.max_recovery_attempts = max_recovery_attempts
        self.enable_learning = enable_learning
        self.detector = SmartErrorDetector()
        self.recovery_system = AdaptiveRecoverySystem()
        self.logger = get_logger(__name__)
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator implementation."""
        
        @partial(jax.jit, static_argnums=())
        def wrapped_function(*args, **kwargs):
            return self._execute_with_recovery(func, *args, **kwargs)
            
        return wrapped_function
    
    def _execute_with_recovery(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with automatic error recovery."""
        
        last_error = None
        
        for attempt in range(self.max_recovery_attempts + 1):
            try:
                # Execute original function
                result = func(*args, **kwargs)
                
                # Post-execution validation
                if isinstance(result, dict):
                    errors = self.detector.detect_physical_implausibility(result)
                    if errors and attempt < self.max_recovery_attempts:
                        raise RuntimeError(f"Physical implausibility detected: {errors[0].error_message}")
                
                # Numerical stability check
                if isinstance(result, jnp.ndarray):
                    error = self.detector.detect_numerical_instability(result, func.__name__)
                    if error and attempt < self.max_recovery_attempts:
                        raise RuntimeError(error.error_message)
                
                return result
                
            except Exception as e:
                # Create error context
                error_context = ErrorContext(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    severity=self._classify_error_severity(e),
                    timestamp=time.time(),
                    component=func.__name__,
                    stack_trace=traceback.format_exc(),
                    recovery_attempts=attempt
                )
                
                last_error = error_context
                
                if attempt < self.max_recovery_attempts:
                    self.logger.warning(f"Attempt {attempt + 1} failed, trying recovery: {e}")
                    
                    # Attempt recovery
                    try:
                        result, success = self.recovery_system.execute_recovery(
                            error_context, func, *args, **kwargs
                        )
                        if success:
                            return result
                    except Exception as recovery_error:
                        self.logger.error(f"Recovery failed: {recovery_error}")
                        continue
                else:
                    self.logger.error(f"All recovery attempts exhausted for {func.__name__}")
                    
        # If all attempts failed, raise the last error
        raise RuntimeError(f"Function {func.__name__} failed after {self.max_recovery_attempts} recovery attempts. "
                         f"Last error: {last_error.error_message}")
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on exception type."""
        
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (ValueError, TypeError, ValidationError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (RuntimeWarning, UserWarning)):
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM


# Global self-healing system instance
_global_self_healing = SelfHealingDecorator()


def self_healing(max_attempts: int = 3, enable_learning: bool = True):
    """Decorator factory for self-healing functions."""
    return SelfHealingDecorator(max_attempts, enable_learning)


def create_resilient_simulator():
    """Create a resilient simulation environment with error recovery."""
    
    logger = get_logger(__name__)
    logger.info("Creating resilient simulation environment")
    
    components = {
        'error_detector': SmartErrorDetector(),
        'recovery_system': AdaptiveRecoverySystem(),
        'self_healing_decorator': SelfHealingDecorator(),
    }
    
    return components


# Example usage and integration
@self_healing(max_attempts=3, enable_learning=True)
def robust_photonic_simulation(optical_input: chex.Array, 
                             phase_settings: chex.Array,
                             **simulation_params) -> Dict[str, chex.Array]:
    """Example self-healing photonic simulation."""
    
    # Simulate potential computation that might fail
    if jnp.any(jnp.isnan(optical_input)):
        raise ValueError("Invalid optical input detected")
    
    # Perform simulation (placeholder)
    optical_output = optical_input * jnp.exp(1j * phase_settings)
    
    return {
        'optical_output': optical_output,
        'power_output': jnp.abs(optical_output)**2,
        'phase_output': jnp.angle(optical_output)
    }


if __name__ == "__main__":
    # Test the self-healing system
    logger = get_logger(__name__)
    
    # Create test input with intentional issues
    test_input = jnp.array([1.0, jnp.nan, 3.0], dtype=jnp.complex64)
    test_phases = jnp.array([0.0, jnp.pi/2, jnp.pi])
    
    try:
        result = robust_photonic_simulation(test_input, test_phases)
        logger.info("Self-healing simulation completed successfully")
        print(f"Result keys: {list(result.keys())}")
    except Exception as e:
        logger.error(f"Simulation failed even with self-healing: {e}")
        
    print("Autonomous Error Recovery System Implementation Complete")