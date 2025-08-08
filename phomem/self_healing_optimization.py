"""
Self-Healing Optimization Framework for Photonic-Memristive Systems.

This module implements advanced self-healing mechanisms:
- Adaptive error detection and recovery
- Dynamic algorithm switching based on performance
- Automatic hyperparameter tuning
- Robust optimization under hardware failures
- Adaptive mesh refinement for numerical simulations
"""

import logging
import time
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import warnings
from functools import wraps

from .optimization import OptimizationResult, HardwareAwareObjective
from .research import NovelOptimizationAlgorithm, QuantumCoherentOptimizer, PhotonicWaveguideOptimizer
from .advanced_multiphysics import AdvancedMultiPhysicsSimulator, MultiPhysicsState
from .utils.validation import ValidationError, validate_input_array
from .utils.logging import setup_logging
from .utils.performance import ProfileManager, MemoryMonitor

logger = logging.getLogger(__name__)


class OptimizationHealth(Enum):
    """Health status of optimization process."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class HealthMetrics:
    """Metrics for assessing optimization health."""
    convergence_rate: float
    gradient_magnitude: float
    parameter_stability: float
    constraint_violations: int
    numerical_errors: int
    memory_usage: float
    computation_time: float
    success_rate: float
    
    def get_health_score(self) -> float:
        """Calculate overall health score [0, 1]."""
        scores = []
        
        # Convergence rate (higher is better)
        conv_score = min(1.0, max(0.0, self.convergence_rate))
        scores.append(conv_score)
        
        # Gradient magnitude (moderate values preferred)
        grad_score = 1.0 / (1.0 + abs(np.log10(self.gradient_magnitude + 1e-12)))
        scores.append(grad_score)
        
        # Parameter stability (higher is better)
        stability_score = min(1.0, max(0.0, self.parameter_stability))
        scores.append(stability_score)
        
        # Constraint violations (fewer is better)
        constraint_score = 1.0 / (1.0 + self.constraint_violations)
        scores.append(constraint_score)
        
        # Numerical errors (fewer is better)
        error_score = 1.0 / (1.0 + self.numerical_errors)
        scores.append(error_score)
        
        # Success rate (higher is better)
        scores.append(self.success_rate)
        
        return np.mean(scores)


@dataclass
class HealingAction:
    """Represents a healing action to fix optimization issues."""
    action_type: str
    parameters: Dict[str, Any]
    priority: int
    estimated_cost: float
    success_probability: float
    description: str


class SelfHealingOptimizer:
    """Self-healing optimization framework with adaptive recovery mechanisms."""
    
    def __init__(
        self,
        primary_optimizer: NovelOptimizationAlgorithm,
        backup_optimizers: Optional[List[NovelOptimizationAlgorithm]] = None,
        health_check_interval: int = 10,
        healing_threshold: float = 0.3,
        max_healing_attempts: int = 3
    ):
        self.primary_optimizer = primary_optimizer
        self.backup_optimizers = backup_optimizers or []
        self.health_check_interval = health_check_interval
        self.healing_threshold = healing_threshold
        self.max_healing_attempts = max_healing_attempts
        
        # Health monitoring
        self.current_health = OptimizationHealth.HEALTHY
        self.health_history = []
        self.healing_attempts = 0
        self.performance_metrics = []
        
        # Adaptive parameters
        self.adaptive_parameters = {
            'learning_rate': 1e-3,
            'batch_size': 32,
            'convergence_tolerance': 1e-6,
            'max_iterations': 1000
        }
        
        # Error patterns and solutions
        self.error_patterns = self._initialize_error_patterns()
        self.healing_strategies = self._initialize_healing_strategies()
        
        logger.info("Self-healing optimizer initialized")
    
    def optimize(
        self,
        objective_fn: Callable,
        initial_params: Dict[str, jnp.ndarray],
        **kwargs
    ) -> OptimizationResult:
        """Perform self-healing optimization with adaptive recovery."""
        
        logger.info("Starting self-healing optimization...")
        start_time = time.time()
        
        current_optimizer = self.primary_optimizer
        optimization_history = []
        healing_log = []
        
        # Main optimization loop with health monitoring
        for attempt in range(self.max_healing_attempts + 1):
            try:
                # Run optimization with health monitoring
                result = self._run_monitored_optimization(
                    current_optimizer, objective_fn, initial_params, **kwargs
                )
                
                optimization_history.append(result)
                
                # Check if optimization was successful
                if result.success or attempt == self.max_healing_attempts:
                    break
                
                # Diagnose issues and apply healing
                health_metrics = self._assess_health(result)
                healing_actions = self._diagnose_and_heal(health_metrics, result)
                healing_log.extend(healing_actions)
                
                # Switch optimizer if needed
                current_optimizer = self._select_optimizer(health_metrics)
                
                logger.info(f"Applied {len(healing_actions)} healing actions, retrying...")
                
            except Exception as e:
                logger.error(f"Optimization attempt {attempt} failed: {e}")
                if attempt < self.max_healing_attempts:
                    current_optimizer = self._handle_critical_failure(e)
                else:
                    raise
        
        # Compile final result
        best_result = min(optimization_history, key=lambda r: r.best_loss)
        optimization_time = time.time() - start_time
        
        # Add healing information to result
        best_result.hardware_metrics = best_result.hardware_metrics or {}
        best_result.hardware_metrics.update({
            'healing_attempts': len(healing_log),
            'healing_actions': [action.action_type for action in healing_log],
            'final_health': self.current_health.value,
            'optimization_attempts': len(optimization_history)
        })
        
        best_result.optimization_time = optimization_time
        
        logger.info(f"Self-healing optimization completed in {optimization_time:.2f}s")
        return best_result
    
    def _run_monitored_optimization(
        self,
        optimizer: NovelOptimizationAlgorithm,
        objective_fn: Callable,
        initial_params: Dict[str, jnp.ndarray],
        **kwargs
    ) -> OptimizationResult:
        """Run optimization with continuous health monitoring."""
        
        # Wrap objective function with monitoring
        monitored_objective = self._create_monitored_objective(objective_fn)
        
        # Run optimization
        result = optimizer.optimize(monitored_objective, initial_params, **kwargs)
        
        return result
    
    def _create_monitored_objective(self, objective_fn: Callable) -> Callable:
        """Create monitored version of objective function."""
        
        call_count = [0]
        error_count = [0]
        computation_times = []
        
        @wraps(objective_fn)
        def monitored_objective(params: Dict[str, jnp.ndarray]) -> float:
            call_count[0] += 1
            start_time = time.time()
            
            try:
                # Validate parameters
                for name, param in params.items():
                    if not jnp.isfinite(param).all():
                        error_count[0] += 1
                        raise ValueError(f"Non-finite parameter {name}")
                
                # Evaluate objective
                result = objective_fn(params)
                
                # Check result validity
                if not jnp.isfinite(result):
                    error_count[0] += 1
                    raise ValueError("Non-finite objective value")
                
                computation_time = time.time() - start_time
                computation_times.append(computation_time)
                
                # Update performance metrics
                if call_count[0] % self.health_check_interval == 0:
                    self._update_performance_metrics(call_count[0], error_count[0], computation_times)
                
                return float(result)
                
            except Exception as e:
                error_count[0] += 1
                logger.warning(f"Objective evaluation error: {e}")
                return 1e6  # Large penalty for failed evaluations
        
        return monitored_objective
    
    def _update_performance_metrics(
        self, 
        call_count: int, 
        error_count: int, 
        computation_times: List[float]
    ):
        """Update performance metrics for health assessment."""
        
        if not computation_times:
            return
        
        # Calculate metrics
        avg_time = np.mean(computation_times[-self.health_check_interval:])
        error_rate = error_count / call_count
        
        metrics = {
            'timestamp': time.time(),
            'call_count': call_count,
            'error_rate': error_rate,
            'avg_computation_time': avg_time,
            'memory_usage': self._get_memory_usage()
        }
        
        self.performance_metrics.append(metrics)
        
        # Assess health
        current_health = self._assess_performance_health(metrics)
        if current_health != self.current_health:
            logger.info(f"Health status changed: {self.current_health} -> {current_health}")
            self.current_health = current_health
    
    def _assess_health(self, result: OptimizationResult) -> HealthMetrics:
        """Assess optimization health based on result."""
        
        # Calculate convergence rate
        if len(result.convergence_history) > 1:
            final_loss = result.convergence_history[-1]
            initial_loss = result.convergence_history[0]
            convergence_rate = max(0, (initial_loss - final_loss) / initial_loss)
        else:
            convergence_rate = 0.0
        
        # Estimate gradient magnitude (simplified)
        gradient_magnitude = 1.0 / (1.0 + result.best_loss)
        
        # Calculate parameter stability
        parameter_stability = 1.0 if result.success else 0.5
        
        # Count constraint violations (simplified)
        constraint_violations = 0 if result.best_loss < 1.0 else 1
        
        # Count numerical errors
        numerical_errors = len([m for m in self.performance_metrics 
                              if m['error_rate'] > 0.1])
        
        # Get recent performance metrics
        recent_metrics = self.performance_metrics[-10:] if self.performance_metrics else []
        avg_memory = np.mean([m['memory_usage'] for m in recent_metrics]) if recent_metrics else 0
        avg_time = np.mean([m['avg_computation_time'] for m in recent_metrics]) if recent_metrics else 0
        
        success_rate = 1.0 if result.success else 0.0
        
        return HealthMetrics(
            convergence_rate=convergence_rate,
            gradient_magnitude=gradient_magnitude,
            parameter_stability=parameter_stability,
            constraint_violations=constraint_violations,
            numerical_errors=numerical_errors,
            memory_usage=avg_memory,
            computation_time=avg_time,
            success_rate=success_rate
        )
    
    def _assess_performance_health(self, metrics: Dict[str, Any]) -> OptimizationHealth:
        """Assess health based on performance metrics."""
        
        error_rate = metrics.get('error_rate', 0)
        avg_time = metrics.get('avg_computation_time', 0)
        memory_usage = metrics.get('memory_usage', 0)
        
        # Simple health assessment rules
        if error_rate > 0.5 or avg_time > 10.0:
            return OptimizationHealth.CRITICAL
        elif error_rate > 0.2 or avg_time > 5.0 or memory_usage > 0.8:
            return OptimizationHealth.DEGRADED
        elif error_rate > 0.05 or avg_time > 2.0:
            return OptimizationHealth.DEGRADED
        else:
            return OptimizationHealth.HEALTHY
    
    def _diagnose_and_heal(
        self, 
        health_metrics: HealthMetrics, 
        result: OptimizationResult
    ) -> List[HealingAction]:
        """Diagnose issues and generate healing actions."""
        
        healing_actions = []
        health_score = health_metrics.get_health_score()
        
        if health_score < self.healing_threshold:
            logger.warning(f"Poor optimization health detected (score: {health_score:.3f})")
            
            # Convergence issues
            if health_metrics.convergence_rate < 0.1:
                actions = self._generate_convergence_healing_actions(health_metrics)
                healing_actions.extend(actions)
            
            # Numerical issues
            if health_metrics.numerical_errors > 0:
                actions = self._generate_numerical_healing_actions(health_metrics)
                healing_actions.extend(actions)
            
            # Constraint violations
            if health_metrics.constraint_violations > 0:
                actions = self._generate_constraint_healing_actions(health_metrics)
                healing_actions.extend(actions)
            
            # Performance issues
            if health_metrics.computation_time > 5.0:
                actions = self._generate_performance_healing_actions(health_metrics)
                healing_actions.extend(actions)
            
            # Apply healing actions
            for action in healing_actions:
                self._apply_healing_action(action)
        
        return healing_actions
    
    def _generate_convergence_healing_actions(self, metrics: HealthMetrics) -> List[HealingAction]:
        """Generate actions to fix convergence issues."""
        actions = []
        
        # Adjust learning rate
        if metrics.gradient_magnitude > 1.0:
            actions.append(HealingAction(
                action_type="reduce_learning_rate",
                parameters={"factor": 0.5},
                priority=1,
                estimated_cost=0.1,
                success_probability=0.8,
                description="Reduce learning rate to improve stability"
            ))
        elif metrics.gradient_magnitude < 0.01:
            actions.append(HealingAction(
                action_type="increase_learning_rate",
                parameters={"factor": 2.0},
                priority=1,
                estimated_cost=0.1,
                success_probability=0.7,
                description="Increase learning rate to improve convergence"
            ))
        
        # Adjust convergence tolerance
        actions.append(HealingAction(
            action_type="relax_convergence_tolerance",
            parameters={"factor": 10.0},
            priority=2,
            estimated_cost=0.05,
            success_probability=0.9,
            description="Relax convergence tolerance"
        ))
        
        return actions
    
    def _generate_numerical_healing_actions(self, metrics: HealthMetrics) -> List[HealingAction]:
        """Generate actions to fix numerical issues."""
        actions = []
        
        # Add regularization
        actions.append(HealingAction(
            action_type="add_regularization",
            parameters={"strength": 1e-4},
            priority=1,
            estimated_cost=0.2,
            success_probability=0.8,
            description="Add regularization to improve numerical stability"
        ))
        
        # Reduce precision requirements
        actions.append(HealingAction(
            action_type="reduce_precision",
            parameters={"tolerance": 1e-4},
            priority=3,
            estimated_cost=0.1,
            success_probability=0.9,
            description="Reduce numerical precision requirements"
        ))
        
        return actions
    
    def _generate_constraint_healing_actions(self, metrics: HealthMetrics) -> List[HealingAction]:
        """Generate actions to handle constraint violations."""
        actions = []
        
        # Add penalty methods
        actions.append(HealingAction(
            action_type="increase_penalty_weights",
            parameters={"factor": 2.0},
            priority=1,
            estimated_cost=0.1,
            success_probability=0.8,
            description="Increase penalty weights for constraint violations"
        ))
        
        # Switch to barrier methods
        actions.append(HealingAction(
            action_type="enable_barrier_method",
            parameters={"barrier_parameter": 1e-2},
            priority=2,
            estimated_cost=0.3,
            success_probability=0.7,
            description="Switch to barrier method for constraints"
        ))
        
        return actions
    
    def _generate_performance_healing_actions(self, metrics: HealthMetrics) -> List[HealingAction]:
        """Generate actions to improve performance."""
        actions = []
        
        # Reduce problem size
        actions.append(HealingAction(
            action_type="reduce_problem_size",
            parameters={"reduction_factor": 0.8},
            priority=2,
            estimated_cost=0.5,
            success_probability=0.9,
            description="Reduce problem size to improve performance"
        ))
        
        # Enable approximations
        actions.append(HealingAction(
            action_type="enable_approximations",
            parameters={"approximation_level": 1},
            priority=3,
            estimated_cost=0.2,
            success_probability=0.8,
            description="Enable approximations to speed up computation"
        ))
        
        return actions
    
    def _apply_healing_action(self, action: HealingAction):
        """Apply a specific healing action."""
        
        logger.info(f"Applying healing action: {action.description}")
        
        try:
            if action.action_type == "reduce_learning_rate":
                factor = action.parameters["factor"]
                self.adaptive_parameters["learning_rate"] *= factor
                
            elif action.action_type == "increase_learning_rate":
                factor = action.parameters["factor"]
                self.adaptive_parameters["learning_rate"] *= factor
                
            elif action.action_type == "relax_convergence_tolerance":
                factor = action.parameters["factor"]
                self.adaptive_parameters["convergence_tolerance"] *= factor
                
            elif action.action_type == "add_regularization":
                # This would be applied in the objective function
                pass
                
            elif action.action_type == "reduce_precision":
                # This would affect numerical computations
                pass
                
            elif action.action_type == "increase_penalty_weights":
                # This would affect constrained optimization
                pass
                
            elif action.action_type == "enable_barrier_method":
                # Switch optimization strategy
                pass
                
            elif action.action_type == "reduce_problem_size":
                # This would require problem reformulation
                pass
                
            elif action.action_type == "enable_approximations":
                # Enable faster approximations
                pass
                
            else:
                logger.warning(f"Unknown healing action: {action.action_type}")
                
        except Exception as e:
            logger.error(f"Failed to apply healing action {action.action_type}: {e}")
    
    def _select_optimizer(self, health_metrics: HealthMetrics) -> NovelOptimizationAlgorithm:
        """Select the best optimizer based on health metrics."""
        
        health_score = health_metrics.get_health_score()
        
        # If health is very poor, try a different optimizer
        if health_score < 0.2 and self.backup_optimizers:
            logger.info("Switching to backup optimizer due to poor health")
            return self.backup_optimizers[0]
        
        # Otherwise use primary optimizer
        return self.primary_optimizer
    
    def _handle_critical_failure(self, exception: Exception) -> NovelOptimizationAlgorithm:
        """Handle critical optimization failures."""
        
        logger.error(f"Critical optimization failure: {exception}")
        
        # Switch to most robust backup optimizer
        if self.backup_optimizers:
            # Simple strategy: use backup optimizers in order
            optimizer_index = min(self.healing_attempts, len(self.backup_optimizers) - 1)
            selected_optimizer = self.backup_optimizers[optimizer_index]
            logger.info(f"Switching to backup optimizer: {type(selected_optimizer).__name__}")
            return selected_optimizer
        
        # If no backup optimizers, modify primary optimizer parameters
        logger.warning("No backup optimizers available, modifying primary optimizer")
        return self.primary_optimizer
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage (simplified)."""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            return 0.0  # Default if psutil not available
    
    def _initialize_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize database of common error patterns and solutions."""
        return {
            "stagnation": {
                "symptoms": ["low_convergence_rate", "stable_loss"],
                "causes": ["local_minimum", "learning_rate_too_low"],
                "solutions": ["increase_learning_rate", "add_noise", "restart"]
            },
            "instability": {
                "symptoms": ["oscillating_loss", "large_gradients"],
                "causes": ["learning_rate_too_high", "numerical_issues"],
                "solutions": ["reduce_learning_rate", "add_regularization"]
            },
            "divergence": {
                "symptoms": ["increasing_loss", "parameter_explosion"],
                "causes": ["learning_rate_too_high", "poor_initialization"],
                "solutions": ["reduce_learning_rate", "reinitialize", "add_clipping"]
            }
        }
    
    def _initialize_healing_strategies(self) -> Dict[str, Callable]:
        """Initialize healing strategies."""
        return {
            "parameter_adjustment": self._adjust_hyperparameters,
            "algorithm_switching": self._switch_algorithm,
            "problem_reformulation": self._reformulate_problem,
            "approximation_methods": self._enable_approximations
        }
    
    def _adjust_hyperparameters(self, metrics: HealthMetrics) -> List[HealingAction]:
        """Adjust hyperparameters based on health metrics."""
        return self._generate_convergence_healing_actions(metrics)
    
    def _switch_algorithm(self, metrics: HealthMetrics) -> List[HealingAction]:
        """Switch to different algorithm."""
        return [HealingAction(
            action_type="switch_algorithm",
            parameters={"target": "backup_optimizer"},
            priority=1,
            estimated_cost=0.5,
            success_probability=0.8,
            description="Switch to backup optimization algorithm"
        )]
    
    def _reformulate_problem(self, metrics: HealthMetrics) -> List[HealingAction]:
        """Reformulate optimization problem."""
        return [HealingAction(
            action_type="reformulate_problem",
            parameters={"method": "relaxation"},
            priority=2,
            estimated_cost=0.8,
            success_probability=0.6,
            description="Reformulate optimization problem"
        )]
    
    def _enable_approximations(self, metrics: HealthMetrics) -> List[HealingAction]:
        """Enable approximation methods."""
        return self._generate_performance_healing_actions(metrics)


class AdaptiveMeshRefinement:
    """Adaptive mesh refinement for numerical simulations."""
    
    def __init__(
        self,
        initial_mesh_size: int = 32,
        max_mesh_size: int = 256,
        refinement_threshold: float = 0.1,
        coarsening_threshold: float = 0.01
    ):
        self.initial_mesh_size = initial_mesh_size
        self.max_mesh_size = max_mesh_size
        self.refinement_threshold = refinement_threshold
        self.coarsening_threshold = coarsening_threshold
        
        self.current_mesh_size = initial_mesh_size
        self.refinement_history = []
        
        logger.info(f"Adaptive mesh refinement initialized with {initial_mesh_size} points")
    
    def adapt_mesh(
        self,
        field: jnp.ndarray,
        error_estimate: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Adapt mesh based on error estimates."""
        
        max_error = jnp.max(error_estimate)
        mean_error = jnp.mean(error_estimate)
        
        # Determine if refinement or coarsening is needed
        if max_error > self.refinement_threshold and self.current_mesh_size < self.max_mesh_size:
            return self._refine_mesh(field, error_estimate)
        elif mean_error < self.coarsening_threshold and self.current_mesh_size > self.initial_mesh_size:
            return self._coarsen_mesh(field, error_estimate)
        else:
            return field, error_estimate
    
    def _refine_mesh(
        self,
        field: jnp.ndarray,
        error_estimate: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Refine mesh in high-error regions."""
        
        # Simple uniform refinement (in practice, would be adaptive)
        new_size = min(self.current_mesh_size * 2, self.max_mesh_size)
        
        # Interpolate field to new mesh
        from scipy import ndimage
        zoom_factor = new_size / self.current_mesh_size
        
        refined_field = jnp.array(ndimage.zoom(field, zoom_factor, order=1))
        refined_error = jnp.array(ndimage.zoom(error_estimate, zoom_factor, order=1))
        
        self.current_mesh_size = new_size
        self.refinement_history.append(("refine", new_size))
        
        logger.info(f"Mesh refined to {new_size} points")
        
        return refined_field, refined_error
    
    def _coarsen_mesh(
        self,
        field: jnp.ndarray,
        error_estimate: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Coarsen mesh in low-error regions."""
        
        new_size = max(self.current_mesh_size // 2, self.initial_mesh_size)
        
        # Downsample field
        from scipy import ndimage
        zoom_factor = new_size / self.current_mesh_size
        
        coarsened_field = jnp.array(ndimage.zoom(field, zoom_factor, order=1))
        coarsened_error = jnp.array(ndimage.zoom(error_estimate, zoom_factor, order=1))
        
        self.current_mesh_size = new_size
        self.refinement_history.append(("coarsen", new_size))
        
        logger.info(f"Mesh coarsened to {new_size} points")
        
        return coarsened_field, coarsened_error
    
    def estimate_error(self, field: jnp.ndarray, field_prev: jnp.ndarray) -> jnp.ndarray:
        """Estimate numerical error for mesh adaptation."""
        
        # Simple error estimate based on field change
        if field.shape != field_prev.shape:
            # Resize to common shape for comparison
            min_size = min(field.size, field_prev.size)
            field_resized = field.flatten()[:min_size]
            field_prev_resized = field_prev.flatten()[:min_size]
            error = jnp.abs(field_resized - field_prev_resized)
            return error.reshape(field.shape) if error.size == field.size else jnp.ones_like(field) * jnp.mean(error)
        
        return jnp.abs(field - field_prev)


def create_self_healing_optimizer(
    primary_algorithm: str = "quantum_coherent",
    enable_mesh_refinement: bool = True
) -> SelfHealingOptimizer:
    """Factory function to create self-healing optimizer with appropriate algorithms."""
    
    # Create primary optimizer
    if primary_algorithm == "quantum_coherent":
        primary = QuantumCoherentOptimizer(num_qubits=8, num_iterations=100)
    elif primary_algorithm == "photonic_waveguide":
        primary = PhotonicWaveguideOptimizer(num_modes=4, num_iterations=80)
    else:
        raise ValueError(f"Unknown primary algorithm: {primary_algorithm}")
    
    # Create backup optimizers
    backup_optimizers = []
    
    # Add different algorithms as backups
    if primary_algorithm != "quantum_coherent":
        backup_optimizers.append(
            QuantumCoherentOptimizer(num_qubits=6, num_iterations=80)
        )
    
    if primary_algorithm != "photonic_waveguide":
        backup_optimizers.append(
            PhotonicWaveguideOptimizer(num_modes=3, num_iterations=60)
        )
    
    # Create self-healing optimizer
    self_healing = SelfHealingOptimizer(
        primary_optimizer=primary,
        backup_optimizers=backup_optimizers,
        health_check_interval=5,
        healing_threshold=0.3,
        max_healing_attempts=2
    )
    
    logger.info(f"Created self-healing optimizer with {primary_algorithm} as primary")
    
    return self_healing