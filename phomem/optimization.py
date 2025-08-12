"""
Advanced optimization algorithms for photonic-memristive neural networks.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from scipy.optimize import differential_evolution, basinhopping
import matplotlib.pyplot as plt

from .utils.validation import ValidationError, validate_input_array
from .utils.logging import setup_logging
from .utils.performance import PerformanceOptimizer, MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from optimization process."""
    best_params: Dict[str, jnp.ndarray]
    best_loss: float
    convergence_history: List[float]
    optimization_time: float
    iterations: int
    success: bool
    final_gradient_norm: Optional[float] = None
    hardware_metrics: Optional[Dict[str, float]] = None
    pareto_front: Optional[List[Tuple[float, ...]]] = None


class HardwareAwareObjective(ABC):
    """Abstract base class for hardware-aware optimization objectives."""
    
    @abstractmethod
    def compute_loss(self, params: Dict[str, jnp.ndarray], inputs: jnp.ndarray, targets: jnp.ndarray) -> float:
        """Compute primary loss (accuracy/task performance)."""
        pass
    
    @abstractmethod
    def compute_hardware_penalties(self, params: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """Compute hardware-specific penalty terms."""
        pass
    
    def compute_total_objective(
        self, 
        params: Dict[str, jnp.ndarray], 
        inputs: jnp.ndarray, 
        targets: jnp.ndarray,
        penalty_weights: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """Compute total multi-objective loss."""
        
        # Primary task loss
        task_loss = self.compute_loss(params, inputs, targets)
        
        # Hardware penalties
        penalties = self.compute_hardware_penalties(params)
        
        # Weighted combination
        total_loss = task_loss
        for penalty_name, penalty_value in penalties.items():
            weight = penalty_weights.get(penalty_name, 0.0)
            total_loss += weight * penalty_value
        
        # Return total loss and breakdown
        loss_breakdown = {
            'task_loss': task_loss,
            'total_loss': total_loss,
            **penalties
        }
        
        return total_loss, loss_breakdown


class PhotonicMemristiveObjective(HardwareAwareObjective):
    """Hardware-aware objective for photonic-memristive networks."""
    
    def __init__(self, network, device_constraints: Dict[str, Any]):
        self.network = network
        self.device_constraints = device_constraints
        
        # Hardware limits
        self.max_optical_power = device_constraints.get('max_optical_power', 1.0)  # W
        self.max_phase_shift = device_constraints.get('max_phase_shift', 2*np.pi)
        self.max_memristor_current = device_constraints.get('max_memristor_current', 1e-3)  # A
        self.max_temperature = device_constraints.get('max_temperature', 400)  # K
        self.min_resistance = device_constraints.get('min_resistance', 1e3)  # Ω
        self.max_resistance = device_constraints.get('max_resistance', 1e8)  # Ω
        
    def compute_loss(self, params: Dict[str, jnp.ndarray], inputs: jnp.ndarray, targets: jnp.ndarray) -> float:
        """Compute task-specific loss (MSE for regression, CE for classification)."""
        try:
            predictions = self.network.apply(params, inputs)
            
            # Handle different output formats
            if predictions.ndim != targets.ndim:
                if predictions.ndim == 2 and targets.ndim == 1:
                    # Classification case - take argmax
                    predictions = jnp.argmax(predictions, axis=1)
                elif predictions.ndim == 1 and targets.ndim == 2:
                    # Regression case - reshape
                    predictions = predictions.reshape(targets.shape)
            
            # Compute appropriate loss
            if targets.dtype == jnp.int32 or targets.dtype == jnp.int64:
                # Classification - cross-entropy
                if predictions.ndim == 1:
                    # Convert to one-hot for proper CE calculation
                    num_classes = jnp.max(targets) + 1
                    targets_onehot = jnp.eye(num_classes)[targets]
                    predictions_onehot = jnp.eye(num_classes)[predictions]
                    loss = -jnp.mean(jnp.sum(targets_onehot * jnp.log(predictions_onehot + 1e-8), axis=1))
                else:
                    loss = optax.softmax_cross_entropy(predictions, targets).mean()
            else:
                # Regression - MSE
                loss = jnp.mean((predictions - targets) ** 2)
            
            return float(loss)
            
        except Exception as e:
            logger.error(f"Error computing task loss: {e}")
            return 1e6  # Large penalty for invalid configurations
    
    def compute_hardware_penalties(self, params: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """Compute hardware-specific penalties."""
        penalties = {}
        
        try:
            # Optical power penalty
            optical_power = self._estimate_optical_power(params)
            if optical_power > self.max_optical_power:
                penalties['optical_power'] = (optical_power - self.max_optical_power)**2
            else:
                penalties['optical_power'] = 0.0
            
            # Phase shift constraints
            phase_penalty = self._compute_phase_penalties(params)
            penalties['phase_constraints'] = phase_penalty
            
            # Memristor constraints
            memristor_penalty = self._compute_memristor_penalties(params)
            penalties['memristor_constraints'] = memristor_penalty
            
            # Thermal penalty
            thermal_penalty = self._compute_thermal_penalty(params)
            penalties['thermal'] = thermal_penalty
            
            # Device variability penalty
            variability_penalty = self._compute_variability_penalty(params)
            penalties['variability'] = variability_penalty
            
            # Aging penalty
            aging_penalty = self._compute_aging_penalty(params)
            penalties['aging'] = aging_penalty
            
        except Exception as e:
            logger.error(f"Error computing hardware penalties: {e}")
            # Return large penalties for invalid parameters
            penalties = {
                'optical_power': 1e3,
                'phase_constraints': 1e3,
                'memristor_constraints': 1e3,
                'thermal': 1e3,
                'variability': 1e3,
                'aging': 1e3
            }
        
        return penalties
    
    def _estimate_optical_power(self, params: Dict[str, jnp.ndarray]) -> float:
        """Estimate total optical power consumption."""
        total_power = 0.0
        
        for param_name, param_values in params.items():
            if 'phase_shift' in param_name:
                # Thermal phase shifters: Power ∝ phase_shift²
                phase_shifts = param_values.flatten()
                power_per_shifter = 20e-3 * (phase_shifts / np.pi)**2  # 20mW for π shift
                total_power += jnp.sum(power_per_shifter)
        
        return float(total_power)
    
    def _compute_phase_penalties(self, params: Dict[str, jnp.ndarray]) -> float:
        """Compute penalty for phase shift constraints."""
        penalty = 0.0
        
        for param_name, param_values in params.items():
            if 'phase_shift' in param_name:
                # Penalty for phase shifts exceeding 2π
                excess_phase = jnp.maximum(0, jnp.abs(param_values) - self.max_phase_shift)
                penalty += jnp.sum(excess_phase**2)
        
        return float(penalty)
    
    def _compute_memristor_penalties(self, params: Dict[str, jnp.ndarray]) -> float:
        """Compute penalty for memristor constraints."""
        penalty = 0.0
        
        for param_name, param_values in params.items():
            if 'resistance' in param_name or 'conductance' in param_name:
                if 'resistance' in param_name:
                    resistance = param_values
                else:
                    # Convert conductance to resistance
                    resistance = 1.0 / (param_values + 1e-12)
                
                # Penalty for resistance outside valid range
                low_penalty = jnp.maximum(0, self.min_resistance - resistance)
                high_penalty = jnp.maximum(0, resistance - self.max_resistance)
                penalty += jnp.sum(low_penalty**2 + high_penalty**2)
                
                # Penalty for very high current (V²/R, assuming unit voltage)
                current = 1.0 / resistance
                excess_current = jnp.maximum(0, current - self.max_memristor_current)
                penalty += jnp.sum(excess_current**2)
        
        return float(penalty)
    
    def _compute_thermal_penalty(self, params: Dict[str, jnp.ndarray]) -> float:
        """Compute penalty for thermal effects."""
        penalty = 0.0
        
        # Estimate temperature rise from power dissipation
        optical_power = self._estimate_optical_power(params)
        
        # Simple thermal model: ΔT = P * R_th
        thermal_resistance = 100  # K/W (typical for integrated photonics)
        temperature_rise = optical_power * thermal_resistance
        ambient_temp = 300  # K
        device_temperature = ambient_temp + temperature_rise
        
        if device_temperature > self.max_temperature:
            penalty = (device_temperature - self.max_temperature)**2
        
        return float(penalty)
    
    def _compute_variability_penalty(self, params: Dict[str, jnp.ndarray]) -> float:
        """Compute penalty for device variability sensitivity."""
        penalty = 0.0
        
        # Estimate sensitivity to parameter variations
        # High sensitivity indicates poor manufacturability
        for param_name, param_values in params.items():
            # Compute local gradient (finite difference approximation)
            param_std = jnp.std(param_values)
            param_range = jnp.max(param_values) - jnp.min(param_values)
            
            if param_range > 0:
                # Penalty for high dynamic range (harder to fabricate)
                dynamic_range = param_range / (jnp.abs(jnp.mean(param_values)) + 1e-8)
                penalty += dynamic_range**2
        
        return float(penalty * 1e-3)  # Scale down this penalty
    
    def _compute_aging_penalty(self, params: Dict[str, jnp.ndarray]) -> float:
        """Compute penalty for device aging effects."""
        penalty = 0.0
        
        for param_name, param_values in params.items():
            if 'resistance' in param_name:
                # Memristor aging depends on switching frequency and amplitude
                resistance = param_values
                
                # Higher resistance contrast leads to more aging
                resistance_ratio = jnp.max(resistance) / (jnp.min(resistance) + 1e-12)
                aging_factor = jnp.log(resistance_ratio + 1)
                penalty += aging_factor
        
        return float(penalty * 1e-4)  # Scale down this penalty


class MultiObjectiveOptimizer:
    """Multi-objective optimizer for hardware-aware neural network design."""
    
    def __init__(
        self,
        objective: HardwareAwareObjective,
        penalty_weights: Optional[Dict[str, float]] = None,
        optimization_method: str = 'adam'
    ):
        self.objective = objective
        self.penalty_weights = penalty_weights or {
            'optical_power': 0.1,
            'phase_constraints': 1.0,
            'memristor_constraints': 1.0,
            'thermal': 0.01,
            'variability': 0.001,
            'aging': 0.0001
        }
        self.optimization_method = optimization_method
        
        # Setup optimizer
        if optimization_method == 'adam':
            self.optimizer = optax.adam(learning_rate=1e-3)
        elif optimization_method == 'sgd':
            self.optimizer = optax.sgd(learning_rate=1e-2)
        elif optimization_method == 'adamw':
            self.optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")
    
    def optimize(
        self,
        initial_params: Dict[str, jnp.ndarray],
        train_data: Tuple[jnp.ndarray, jnp.ndarray],
        val_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        num_epochs: int = 100,
        early_stopping: bool = True,
        patience: int = 10
    ) -> OptimizationResult:
        """Perform multi-objective optimization."""
        
        logger.info(f"Starting multi-objective optimization with {self.optimization_method}")
        start_time = time.time()
        
        train_inputs, train_targets = train_data
        if val_data is not None:
            val_inputs, val_targets = val_data
        else:
            val_inputs, val_targets = train_inputs, train_targets
        
        # Initialize optimizer state
        opt_state = self.optimizer.init(initial_params)
        current_params = initial_params
        
        # Tracking
        train_losses = []
        val_losses = []
        hardware_metrics = []
        best_val_loss = float('inf')
        best_params = initial_params
        patience_counter = 0
        
        # JIT compile loss and gradient functions
        @jax.jit
        def compute_loss_and_grads(params, inputs, targets):
            def loss_fn(params):
                total_loss, breakdown = self.objective.compute_total_objective(
                    params, inputs, targets, self.penalty_weights
                )
                return total_loss, breakdown
            
            (loss, breakdown), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            return loss, breakdown, grads
        
        # Training loop
        for epoch in range(num_epochs):
            # Training step
            train_loss, train_breakdown, grads = compute_loss_and_grads(
                current_params, train_inputs, train_targets
            )
            
            # Update parameters
            updates, opt_state = self.optimizer.update(grads, opt_state, current_params)
            current_params = optax.apply_updates(current_params, updates)
            
            # Validation step
            val_loss, val_breakdown, _ = compute_loss_and_grads(
                current_params, val_inputs, val_targets
            )
            
            # Track progress
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))
            hardware_metrics.append(val_breakdown)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = current_params
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Logging
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                logger.info(
                    f"Epoch {epoch:4d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                    f"task_loss={val_breakdown.get('task_loss', 0):.6f}"
                )
            
            # Early stopping
            if early_stopping and patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Compute final metrics
        optimization_time = time.time() - start_time
        final_loss, final_breakdown, final_grads = compute_loss_and_grads(
            best_params, val_inputs, val_targets
        )
        
        # Compute gradient norm
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_leaves(final_grads)))
        
        result = OptimizationResult(
            best_params=best_params,
            best_loss=float(best_val_loss),
            convergence_history=val_losses,
            optimization_time=optimization_time,
            iterations=epoch + 1,
            success=grad_norm < 1e-3,
            final_gradient_norm=float(grad_norm),
            hardware_metrics=final_breakdown
        )
        
        logger.info(f"Optimization completed in {optimization_time:.2f}s")
        logger.info(f"Final loss: {result.best_loss:.6f}, gradient norm: {result.final_gradient_norm:.2e}")
        
        return result
    
    def pareto_optimization(
        self,
        initial_params: Dict[str, jnp.ndarray],
        train_data: Tuple[jnp.ndarray, jnp.ndarray],
        objectives: List[str] = ['task_loss', 'optical_power', 'thermal'],
        num_generations: int = 50,
        population_size: int = 20
    ) -> OptimizationResult:
        """Perform Pareto-optimal multi-objective optimization."""
        
        logger.info("Starting Pareto multi-objective optimization")
        start_time = time.time()
        
        train_inputs, train_targets = train_data
        
        # Parameter bounds (for differential evolution)
        param_bounds = []
        param_shapes = []
        param_names = []
        
        for name, param in initial_params.items():
            param_shapes.append(param.shape)
            param_names.append(name)
            flat_param = param.flatten()
            
            # Set bounds based on parameter type
            if 'phase' in name.lower():
                bounds = [(-2*np.pi, 2*np.pi)] * len(flat_param)
            elif 'resistance' in name.lower():
                bounds = [(1e3, 1e8)] * len(flat_param)
            elif 'weight' in name.lower():
                bounds = [(-2.0, 2.0)] * len(flat_param)
            else:
                bounds = [(-1.0, 1.0)] * len(flat_param)
            
            param_bounds.extend(bounds)
        
        def objective_function(flat_params):
            """Objective function for differential evolution."""
            # Reconstruct parameter dictionary
            params = {}
            idx = 0
            for i, name in enumerate(param_names):
                shape = param_shapes[i]
                size = np.prod(shape)
                param_data = flat_params[idx:idx+size].reshape(shape)
                params[name] = jnp.array(param_data)
                idx += size
            
            # Compute objectives
            _, breakdown = self.objective.compute_total_objective(
                params, train_inputs, train_targets, {k: 1.0 for k in self.penalty_weights}
            )
            
            # Return selected objectives for Pareto optimization
            return [breakdown.get(obj, 0) for obj in objectives]
        
        # Flatten initial parameters
        flat_initial = []
        for name in param_names:
            flat_initial.extend(initial_params[name].flatten().tolist())
        
        # Use differential evolution for multi-objective optimization
        # Note: This is a simplified approach - full NSGA-II would be better
        pareto_solutions = []
        
        for weight_combination in self._generate_weight_combinations(len(objectives), num_generations):
            # Weighted objective function
            def weighted_objective(flat_params):
                objectives_values = objective_function(flat_params)
                return sum(w * obj for w, obj in zip(weight_combination, objectives_values))
            
            # Optimize
            try:
                result = differential_evolution(
                    weighted_objective,
                    bounds=param_bounds,
                    maxiter=50,
                    popsize=5,
                    seed=42
                )
                
                if result.success:
                    objective_values = objective_function(result.x)
                    pareto_solutions.append(tuple(objective_values))
                    
            except Exception as e:
                logger.warning(f"Differential evolution failed: {e}")
        
        # Find best solution (closest to origin in objective space)
        if pareto_solutions:
            best_solution_idx = np.argmin([sum(sol) for sol in pareto_solutions])
            best_objectives = pareto_solutions[best_solution_idx]
        else:
            best_objectives = [1e6] * len(objectives)
        
        optimization_time = time.time() - start_time
        
        result = OptimizationResult(
            best_params=initial_params,  # Return initial for now - would need to track best
            best_loss=min([sum(sol) for sol in pareto_solutions]) if pareto_solutions else 1e6,
            convergence_history=[],
            optimization_time=optimization_time,
            iterations=num_generations,
            success=len(pareto_solutions) > 0,
            pareto_front=pareto_solutions
        )
        
        logger.info(f"Pareto optimization completed in {optimization_time:.2f}s")
        logger.info(f"Found {len(pareto_solutions)} Pareto solutions")
        
        return result
    
    def _generate_weight_combinations(self, num_objectives: int, num_combinations: int) -> List[List[float]]:
        """Generate weight combinations for Pareto optimization."""
        if num_objectives == 2:
            # Simple case: linearly spaced weights
            weights = []
            for i in range(num_combinations):
                w1 = i / (num_combinations - 1)
                w2 = 1 - w1
                weights.append([w1, w2])
            return weights
        else:
            # For more objectives, use random weights
            weights = []
            np.random.seed(42)
            for _ in range(num_combinations):
                w = np.random.random(num_objectives)
                w = w / np.sum(w)  # Normalize
                weights.append(w.tolist())
            return weights


class NeuralArchitectureSearch:
    """Neural Architecture Search for photonic-memristive networks."""
    
    def __init__(
        self,
        search_space: Dict[str, Any],
        objective_function: Callable,
        hardware_constraints: Dict[str, Any]
    ):
        self.search_space = search_space
        self.objective_function = objective_function
        self.hardware_constraints = hardware_constraints
        
        # Search history
        self.evaluated_architectures = []
        self.best_architecture = None
        self.best_performance = float('inf')
    
    def search(
        self,
        num_trials: int = 100,
        search_strategy: str = 'random'
    ) -> Dict[str, Any]:
        """Perform neural architecture search."""
        
        logger.info(f"Starting NAS with {search_strategy} strategy")
        start_time = time.time()
        
        for trial in range(num_trials):
            # Sample architecture from search space
            if search_strategy == 'random':
                architecture = self._random_sample()
            elif search_strategy == 'evolutionary':
                architecture = self._evolutionary_sample(trial, num_trials)
            else:
                raise ValueError(f"Unknown search strategy: {search_strategy}")
            
            # Evaluate architecture
            try:
                performance = self._evaluate_architecture(architecture)
                
                # Track best
                if performance < self.best_performance:
                    self.best_performance = performance
                    self.best_architecture = architecture
                
                # Store result
                self.evaluated_architectures.append({
                    'architecture': architecture,
                    'performance': performance,
                    'trial': trial
                })
                
                if trial % 10 == 0:
                    logger.info(f"Trial {trial}: performance={performance:.6f}, best={self.best_performance:.6f}")
                
            except Exception as e:
                logger.warning(f"Architecture evaluation failed in trial {trial}: {e}")
        
        search_time = time.time() - start_time
        
        logger.info(f"NAS completed in {search_time:.2f}s")
        logger.info(f"Best architecture performance: {self.best_performance:.6f}")
        
        return {
            'best_architecture': self.best_architecture,
            'best_performance': self.best_performance,
            'search_time': search_time,
            'num_evaluations': len(self.evaluated_architectures),
            'search_history': self.evaluated_architectures
        }
    
    def _random_sample(self) -> Dict[str, Any]:
        """Randomly sample architecture from search space."""
        architecture = {}
        
        for key, space in self.search_space.items():
            if isinstance(space, list):
                # Discrete choice
                architecture[key] = np.random.choice(space)
            elif isinstance(space, tuple) and len(space) == 2:
                # Continuous range
                low, high = space
                if isinstance(low, int) and isinstance(high, int):
                    architecture[key] = np.random.randint(low, high + 1)
                else:
                    architecture[key] = np.random.uniform(low, high)
            else:
                architecture[key] = space  # Fixed value
        
        return architecture
    
    def _evolutionary_sample(self, trial: int, num_trials: int) -> Dict[str, Any]:
        """Sample architecture using evolutionary strategy."""
        if trial < 10 or len(self.evaluated_architectures) < 5:
            # Random sampling for initial population
            return self._random_sample()
        
        # Select parent architectures based on performance
        sorted_architectures = sorted(
            self.evaluated_architectures,
            key=lambda x: x['performance']
        )
        
        # Select from top 20%
        elite_size = max(1, len(sorted_architectures) // 5)
        parent = np.random.choice(sorted_architectures[:elite_size])['architecture']
        
        # Mutate parent
        mutated = self._mutate_architecture(parent)
        return mutated
    
    def _mutate_architecture(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate an architecture."""
        mutated = architecture.copy()
        
        # Randomly select parameters to mutate
        mutation_rate = 0.3
        for key, value in architecture.items():
            if np.random.random() < mutation_rate:
                space = self.search_space[key]
                
                if isinstance(space, list):
                    # Discrete mutation
                    mutated[key] = np.random.choice(space)
                elif isinstance(space, tuple) and len(space) == 2:
                    # Continuous mutation with noise
                    low, high = space
                    if isinstance(low, int) and isinstance(high, int):
                        # Add random noise (±1)
                        noise = np.random.randint(-1, 2)
                        mutated[key] = np.clip(value + noise, low, high)
                    else:
                        # Add Gaussian noise
                        noise = np.random.normal(0, (high - low) * 0.1)
                        mutated[key] = np.clip(value + noise, low, high)
        
        return mutated
    
    def _evaluate_architecture(self, architecture: Dict[str, Any]) -> float:
        """Evaluate architecture performance."""
        # This would typically involve training a network
        # For now, use a simplified performance model
        
        # Basic performance model based on architecture properties
        performance = 0.0
        
        # Layer complexity penalty
        num_layers = architecture.get('num_layers', 3)
        layer_penalty = (num_layers - 2) * 0.1
        performance += layer_penalty
        
        # Size penalty
        hidden_size = architecture.get('hidden_size', 64)
        size_penalty = (hidden_size / 64 - 1) * 0.05
        performance += size_penalty
        
        # Hardware-specific penalties
        photonic_ratio = architecture.get('photonic_ratio', 0.5)
        
        # Optical power penalty
        optical_penalty = photonic_ratio * hidden_size * 1e-5
        performance += optical_penalty
        
        # Add random noise to simulate training variance
        noise = np.random.normal(0, 0.01)
        performance += noise
        
        return performance
    
    def plot_search_progress(self, save_path: Optional[str] = None):
        """Plot NAS progress over time."""
        if not self.evaluated_architectures:
            logger.warning("No architectures evaluated yet")
            return
        
        trials = [arch['trial'] for arch in self.evaluated_architectures]
        performances = [arch['performance'] for arch in self.evaluated_architectures]
        
        # Running best
        running_best = []
        current_best = float('inf')
        for perf in performances:
            if perf < current_best:
                current_best = perf
            running_best.append(current_best)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.scatter(trials, performances, alpha=0.6, s=20)
        plt.plot(trials, running_best, 'r-', linewidth=2, label='Best so far')
        plt.xlabel('Trial')
        plt.ylabel('Performance')
        plt.title('Architecture Search Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(performances, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(self.best_performance, color='red', linestyle='--', 
                   label=f'Best: {self.best_performance:.4f}')
        plt.xlabel('Performance')
        plt.ylabel('Count')
        plt.title('Performance Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


def create_search_space_photonic_memristive() -> Dict[str, Any]:
    """Create search space for photonic-memristive networks."""
    return {
        'num_layers': [2, 3, 4, 5],
        'hidden_size': (16, 128),
        'photonic_layers': [0, 1, 2],  # Which layers are photonic
        'photonic_ratio': (0.0, 1.0),  # Fraction of photonic components
        'phase_shifter_type': ['thermal', 'plasma'],
        'memristor_type': ['pcm_mushroom', 'rram_hfo2'],
        'activation': ['relu', 'tanh', 'gelu'],
        'learning_rate': (1e-4, 1e-2),
        'batch_size': [16, 32, 64],
        'dropout_rate': (0.0, 0.3)
    }


class HyperparameterOptimizer:
    """Bayesian optimization for hyperparameter tuning."""
    
    def __init__(self, objective_function: Callable, parameter_space: Dict[str, Any]):
        self.objective_function = objective_function
        self.parameter_space = parameter_space
        self.history = []
    
    def optimize(self, n_calls: int = 50) -> Dict[str, Any]:
        """Perform Bayesian optimization."""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
        except ImportError:
            logger.error("scikit-optimize not available. Using random search instead.")
            return self._random_search(n_calls)
        
        # Convert parameter space to scikit-optimize format
        dimensions = []
        param_names = []
        
        for name, space in self.parameter_space.items():
            param_names.append(name)
            
            if isinstance(space, list):
                dimensions.append(Categorical(space, name=name))
            elif isinstance(space, tuple) and len(space) == 2:
                low, high = space
                if isinstance(low, int) and isinstance(high, int):
                    dimensions.append(Integer(low, high, name=name))
                else:
                    dimensions.append(Real(low, high, name=name))
        
        def objective_wrapper(params):
            param_dict = dict(zip(param_names, params))
            return self.objective_function(param_dict)
        
        # Perform optimization
        result = gp_minimize(
            func=objective_wrapper,
            dimensions=dimensions,
            n_calls=n_calls,
            random_state=42
        )
        
        best_params = dict(zip(param_names, result.x))
        
        return {
            'best_params': best_params,
            'best_score': result.fun,
            'n_evaluations': len(result.func_vals),
            'convergence': result.func_vals
        }
    
    def _random_search(self, n_calls: int) -> Dict[str, Any]:
        """Fallback random search if scikit-optimize not available."""
        logger.info("Using random search for hyperparameter optimization")
        
        best_score = float('inf')
        best_params = None
        scores = []
        
        for i in range(n_calls):
            # Random sample from parameter space
            params = {}
            for name, space in self.parameter_space.items():
                if isinstance(space, list):
                    params[name] = np.random.choice(space)
                elif isinstance(space, tuple) and len(space) == 2:
                    low, high = space
                    if isinstance(low, int) and isinstance(high, int):
                        params[name] = np.random.randint(low, high + 1)
                    else:
                        params[name] = np.random.uniform(low, high)
            
            # Evaluate
            score = self.objective_function(params)
            scores.append(score)
            
            if score < best_score:
                best_score = score
                best_params = params
            
            if i % 10 == 0:
                logger.info(f"Random search iteration {i}: best_score={best_score:.6f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'n_evaluations': n_calls,
            'convergence': scores
        }