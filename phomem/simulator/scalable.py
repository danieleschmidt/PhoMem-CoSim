"""
Scalable multi-physics simulation with parallel solvers and optimization.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
import queue
import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum

try:
    import jax
    import jax.numpy as jnp
    from jax import pmap, vmap
    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    JAX_AVAILABLE = False

import numpy as np
from pathlib import Path

from .core import OpticalSolver, ThermalSolver, ElectricalSolver
from ..utils.performance import get_performance_optimizer, ConcurrentSimulator
from ..utils.logging import get_logger, create_simulation_logger
from ..utils.exceptions import SimulationError, ConvergenceError, MultiPhysicsError


class SolverType(Enum):
    OPTICAL = "optical"
    THERMAL = "thermal"  
    ELECTRICAL = "electrical"


@dataclass
class SimulationTask:
    """Represents a simulation subtask."""
    task_id: str
    solver_type: SolverType
    geometry: Dict[str, Any]
    boundary_conditions: Dict[str, Any]
    materials: Dict[str, Any]
    priority: int = 0
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass 
class SimulationResult:
    """Result from a simulation task."""
    task_id: str
    solver_type: SolverType
    results: Dict[str, Any]
    computation_time: float
    memory_usage: int
    success: bool
    error_message: str = None


class ParallelSolverOrchestrator:
    """Orchestrates parallel execution of multiple physics solvers."""
    
    def __init__(self, 
                 max_workers: int = None,
                 use_processes: bool = False,
                 solver_cache_size: int = 100):
        
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.use_processes = use_processes
        self.solver_cache_size = solver_cache_size
        
        self.logger = get_logger('simulator.parallel')
        self.optimizer = get_performance_optimizer()
        
        # Initialize solvers
        self.solvers = {
            SolverType.OPTICAL: OpticalSolver(),
            SolverType.THERMAL: ThermalSolver(),
            SolverType.ELECTRICAL: ElectricalSolver()
        }
        
        # Task management
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks = {}
        self.running_tasks = {}
        
        # Thread pool for task execution
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        self.logger.info(f"Initialized parallel solver with {self.max_workers} workers")
    
    def submit_simulation_batch(self, 
                               tasks: List[SimulationTask],
                               coupling_iterations: int = 5,
                               convergence_tolerance: float = 1e-6) -> Dict[str, SimulationResult]:
        """Submit a batch of coupled simulation tasks."""
        
        # Validate task dependencies
        self._validate_task_dependencies(tasks)
        
        # Create dependency graph
        dependency_graph = self._build_dependency_graph(tasks)
        
        # Initialize results storage
        all_results = {}
        
        # Execute coupled iterations
        for iteration in range(coupling_iterations):
            self.logger.info(f"Starting coupling iteration {iteration + 1}/{coupling_iterations}")
            
            # Submit tasks based on dependency order
            iteration_results = self._execute_iteration(tasks, dependency_graph)
            
            # Check convergence
            if iteration > 0:
                converged = self._check_convergence(
                    all_results.get(iteration - 1, {}),
                    iteration_results,
                    convergence_tolerance
                )
                if converged:
                    self.logger.info(f"Converged after {iteration + 1} iterations")
                    break
            
            all_results[iteration] = iteration_results
        
        # Return final iteration results
        return all_results.get(coupling_iterations - 1, {})
    
    def _execute_iteration(self, 
                          tasks: List[SimulationTask],
                          dependency_graph: Dict[str, List[str]]) -> Dict[str, SimulationResult]:
        """Execute one coupling iteration."""
        
        # Track completion status
        completed = set()
        results = {}
        submitted_futures = {}
        
        # Submit initial tasks (those with no dependencies)
        ready_tasks = [task for task in tasks if not task.dependencies]
        
        for task in ready_tasks:
            future = self.executor.submit(self._execute_single_task, task)
            submitted_futures[future] = task
            self.logger.debug(f"Submitted task {task.task_id}")
        
        # Process completed tasks and submit dependent ones
        while submitted_futures:
            # Wait for at least one task to complete
            completed_futures = as_completed(submitted_futures, timeout=300)
            
            for future in completed_futures:
                task = submitted_futures.pop(future)
                
                try:
                    result = future.result()
                    results[task.task_id] = result
                    completed.add(task.task_id)
                    
                    self.logger.debug(f"Completed task {task.task_id} in {result.computation_time:.4f}s")
                    
                    # Check for newly ready tasks
                    newly_ready = []
                    for remaining_task in tasks:
                        if (remaining_task.task_id not in completed and 
                            remaining_task.task_id not in [t.task_id for t in submitted_futures.values()]):
                            
                            # Check if all dependencies are satisfied
                            if all(dep_id in completed for dep_id in remaining_task.dependencies):
                                newly_ready.append(remaining_task)
                    
                    # Submit newly ready tasks
                    for ready_task in newly_ready:
                        future = self.executor.submit(self._execute_single_task, ready_task)
                        submitted_futures[future] = ready_task
                        self.logger.debug(f"Submitted dependent task {ready_task.task_id}")
                
                except Exception as e:
                    self.logger.error(f"Task {task.task_id} failed: {e}")
                    # Create failed result
                    results[task.task_id] = SimulationResult(
                        task_id=task.task_id,
                        solver_type=task.solver_type,
                        results={},
                        computation_time=0.0,
                        memory_usage=0,
                        success=False,
                        error_message=str(e)
                    )
                    completed.add(task.task_id)
        
        return results
    
    def _execute_single_task(self, task: SimulationTask) -> SimulationResult:
        """Execute a single simulation task."""
        start_time = time.time()
        start_memory = self.optimizer._get_memory_usage()
        
        try:
            # Get appropriate solver
            solver = self.solvers[task.solver_type]
            
            # Execute simulation
            with self.optimizer.performance_timer(f'task_{task.task_id}'):
                results = solver.solve(
                    task.geometry,
                    task.boundary_conditions,
                    task.materials
                )
            
            end_time = time.time()
            end_memory = self.optimizer._get_memory_usage()
            
            return SimulationResult(
                task_id=task.task_id,
                solver_type=task.solver_type,
                results=results,
                computation_time=end_time - start_time,
                memory_usage=end_memory - start_memory,
                success=True
            )
        
        except Exception as e:
            end_time = time.time()
            return SimulationResult(
                task_id=task.task_id,
                solver_type=task.solver_type,
                results={},
                computation_time=end_time - start_time,
                memory_usage=0,
                success=False,
                error_message=str(e)
            )
    
    def _validate_task_dependencies(self, tasks: List[SimulationTask]):
        """Validate that task dependencies form a valid DAG."""
        task_ids = {task.task_id for task in tasks}
        
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    raise SimulationError(
                        f"Task {task.task_id} has unknown dependency: {dep_id}",
                        context={'task_id': task.task_id, 'dependency': dep_id}
                    )
        
        # Check for cycles (simplified)
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id, task_map, visited, rec_stack):
            visited.add(task_id)
            rec_stack.add(task_id)
            
            task = task_map[task_id]
            for dep_id in task.dependencies:
                if dep_id not in visited:
                    if has_cycle(dep_id, task_map, visited, rec_stack):
                        return True
                elif dep_id in rec_stack:
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        task_map = {task.task_id: task for task in tasks}
        
        for task in tasks:
            if task.task_id not in visited:
                if has_cycle(task.task_id, task_map, visited, rec_stack):
                    raise SimulationError(
                        f"Circular dependency detected involving task {task.task_id}",
                        context={'task_id': task.task_id}
                    )
    
    def _build_dependency_graph(self, tasks: List[SimulationTask]) -> Dict[str, List[str]]:
        """Build dependency graph from tasks."""
        graph = {}
        for task in tasks:
            graph[task.task_id] = task.dependencies.copy()
        return graph
    
    def _check_convergence(self, 
                          prev_results: Dict[str, SimulationResult],
                          curr_results: Dict[str, SimulationResult],
                          tolerance: float) -> bool:
        """Check if coupled simulation has converged."""
        
        if not prev_results or not curr_results:
            return False
        
        # Compare key metrics between iterations
        for task_id in curr_results:
            if task_id not in prev_results:
                continue
            
            prev_result = prev_results[task_id]
            curr_result = curr_results[task_id]
            
            if not (prev_result.success and curr_result.success):
                continue
            
            # Compare specific fields based on solver type
            if curr_result.solver_type == SolverType.THERMAL:
                # Compare temperature convergence
                prev_temp = prev_result.results.get('final_temperature')
                curr_temp = curr_result.results.get('final_temperature')
                
                if prev_temp is not None and curr_temp is not None:
                    if hasattr(prev_temp, 'shape') and hasattr(curr_temp, 'shape'):
                        if prev_temp.shape == curr_temp.shape:
                            max_change = jnp.max(jnp.abs(curr_temp - prev_temp))
                            if max_change > tolerance:
                                return False
            
            # Similar checks for other solver types could be added
        
        return True
    
    def shutdown(self):
        """Shutdown the parallel solver."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        self.logger.info("Parallel solver shutdown completed")


class ScalableMultiPhysicsSimulator:
    """Scalable multi-physics simulator with advanced optimization."""
    
    def __init__(self,
                 parallel_workers: int = None,
                 use_gpu_acceleration: bool = True,
                 memory_limit_gb: float = 16.0,
                 cache_enabled: bool = True):
        
        self.parallel_workers = parallel_workers or min(8, mp.cpu_count())
        self.use_gpu_acceleration = use_gpu_acceleration and JAX_AVAILABLE
        self.memory_limit_bytes = int(memory_limit_gb * 1024**3)
        self.cache_enabled = cache_enabled
        
        self.logger = get_logger('simulator.scalable')
        self.optimizer = get_performance_optimizer()
        
        # Initialize parallel orchestrator
        self.orchestrator = ParallelSolverOrchestrator(
            max_workers=self.parallel_workers,
            use_processes=True  # Use processes for better parallelism
        )
        
        # Performance monitoring
        self.simulation_stats = {
            'total_simulations': 0,
            'total_compute_time': 0.0,
            'total_wall_time': 0.0,
            'memory_peak': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.logger.info(f"Initialized scalable simulator with {self.parallel_workers} workers")
    
    def simulate_parameter_sweep(self,
                                base_geometry: Dict[str, Any],
                                base_materials: Dict[str, Any],
                                parameter_ranges: Dict[str, List[Any]],
                                physics_types: List[SolverType] = None,
                                max_concurrent: int = None) -> List[Dict[str, Any]]:
        """Run large-scale parameter sweep simulation."""
        
        physics_types = physics_types or [SolverType.OPTICAL, SolverType.THERMAL, SolverType.ELECTRICAL]
        max_concurrent = max_concurrent or self.parallel_workers * 2
        
        # Generate parameter combinations
        import itertools
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        combinations = list(itertools.product(*param_values))
        
        self.logger.info(f"Starting parameter sweep with {len(combinations)} combinations")
        
        # Create simulation tasks
        all_tasks = []
        for combo_idx, combination in enumerate(combinations):
            param_dict = dict(zip(param_names, combination))
            
            # Create tasks for each physics type
            for physics_type in physics_types:
                task = SimulationTask(
                    task_id=f"sweep_{combo_idx}_{physics_type.value}",
                    solver_type=physics_type,
                    geometry=self._update_dict_with_params(base_geometry, param_dict),
                    boundary_conditions=self._generate_boundary_conditions(physics_type, param_dict),
                    materials=base_materials,
                    priority=combo_idx
                )
                all_tasks.append(task)
        
        # Execute in batches to manage memory
        batch_size = min(max_concurrent, len(all_tasks))
        all_results = []
        
        for i in range(0, len(all_tasks), batch_size):
            batch_tasks = all_tasks[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_tasks) + batch_size - 1)//batch_size}")
            
            batch_results = self.orchestrator.submit_simulation_batch(
                batch_tasks,
                coupling_iterations=1  # No coupling for parameter sweep
            )
            
            # Process results
            for task_id, result in batch_results.items():
                combo_idx = int(task_id.split('_')[1])
                physics_type = task_id.split('_')[2]
                
                result_dict = {
                    'combination_index': combo_idx,
                    'parameters': dict(zip(param_names, combinations[combo_idx])),
                    'physics_type': physics_type,
                    'success': result.success,
                    'computation_time': result.computation_time,
                    'results': result.results if result.success else {},
                    'error': result.error_message
                }
                
                all_results.append(result_dict)
            
            # Memory cleanup between batches
            self.optimizer.cleanup_large_objects()
        
        # Update statistics
        self.simulation_stats['total_simulations'] += len(all_results)
        
        return all_results
    
    def simulate_monte_carlo(self,
                           base_config: Dict[str, Any],
                           variability_config: Dict[str, Dict[str, float]],
                           n_samples: int = 1000,
                           confidence_level: float = 0.95) -> Dict[str, Any]:
        """Run Monte Carlo simulation for variability analysis."""
        
        self.logger.info(f"Starting Monte Carlo simulation with {n_samples} samples")
        
        # Generate parameter samples
        samples = self._generate_monte_carlo_samples(variability_config, n_samples)
        
        # Create simulation tasks
        tasks = []
        for sample_idx, sample_params in enumerate(samples):
            
            # Create tasks for each physics domain
            for physics_type in [SolverType.OPTICAL, SolverType.THERMAL, SolverType.ELECTRICAL]:
                task = SimulationTask(
                    task_id=f"mc_{sample_idx}_{physics_type.value}",
                    solver_type=physics_type,
                    geometry=self._update_dict_with_params(base_config['geometry'], sample_params),
                    boundary_conditions=base_config.get('boundary_conditions', {}),
                    materials=self._update_dict_with_params(base_config['materials'], sample_params),
                    priority=sample_idx
                )
                tasks.append(task)
        
        # Execute with coupling
        results = self.orchestrator.submit_simulation_batch(
            tasks,
            coupling_iterations=3,
            convergence_tolerance=1e-6
        )
        
        # Analyze results statistically
        analysis = self._analyze_monte_carlo_results(results, n_samples, confidence_level)
        
        return {
            'n_samples': n_samples,
            'confidence_level': confidence_level,
            'statistical_analysis': analysis,
            'raw_results': results
        }
    
    def optimize_design(self,
                       objective_function: Callable,
                       design_variables: Dict[str, Tuple[float, float]],
                       constraints: List[Callable] = None,
                       optimization_algorithm: str = 'genetic',
                       max_evaluations: int = 1000) -> Dict[str, Any]:
        """Run design optimization using specified algorithm."""
        
        self.logger.info(f"Starting design optimization with {optimization_algorithm} algorithm")
        
        if optimization_algorithm == 'genetic':
            return self._genetic_algorithm_optimization(
                objective_function, design_variables, constraints, max_evaluations
            )
        elif optimization_algorithm == 'particle_swarm':
            return self._particle_swarm_optimization(
                objective_function, design_variables, constraints, max_evaluations
            )
        else:
            raise SimulationError(f"Unknown optimization algorithm: {optimization_algorithm}")
    
    def _generate_monte_carlo_samples(self, 
                                    variability_config: Dict[str, Dict[str, float]],
                                    n_samples: int) -> List[Dict[str, float]]:
        """Generate Monte Carlo parameter samples."""
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for param_name, var_config in variability_config.items():
                distribution = var_config.get('distribution', 'normal')
                mean = var_config.get('mean', 0.0)
                std = var_config.get('std', 1.0)
                
                if distribution == 'normal':
                    value = np.random.normal(mean, std)
                elif distribution == 'uniform':
                    low = var_config.get('low', mean - std)
                    high = var_config.get('high', mean + std)
                    value = np.random.uniform(low, high)
                elif distribution == 'lognormal':
                    sigma = np.sqrt(np.log(1 + (std/mean)**2))
                    mu = np.log(mean) - sigma**2/2
                    value = np.random.lognormal(mu, sigma)
                else:
                    value = mean  # Fallback to mean
                
                sample[param_name] = value
            
            samples.append(sample)
        
        return samples
    
    def _analyze_monte_carlo_results(self, 
                                   results: Dict[str, SimulationResult],
                                   n_samples: int,
                                   confidence_level: float) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results."""
        
        # Group results by sample index
        samples_data = {}
        for task_id, result in results.items():
            if result.success:
                sample_idx = int(task_id.split('_')[1])
                if sample_idx not in samples_data:
                    samples_data[sample_idx] = {}
                
                physics_type = task_id.split('_')[2]
                samples_data[sample_idx][physics_type] = result.results
        
        # Statistical analysis
        successful_samples = len(samples_data)
        yield_rate = successful_samples / n_samples
        
        # Extract key metrics for statistical analysis
        metrics = {}
        for sample_idx, sample_results in samples_data.items():
            for physics_type, physics_results in sample_results.items():
                for metric_name, metric_value in physics_results.items():
                    if isinstance(metric_value, (int, float)):
                        key = f"{physics_type}_{metric_name}"
                        if key not in metrics:
                            metrics[key] = []
                        metrics[key].append(metric_value)
        
        # Calculate statistics
        statistics = {}
        alpha = 1 - confidence_level
        z_score = 1.96  # For 95% confidence interval
        
        for metric_name, values in metrics.items():
            if len(values) > 1:
                mean_val = np.mean(values)
                std_val = np.std(values)
                sem = std_val / np.sqrt(len(values))
                
                statistics[metric_name] = {
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'confidence_interval': [
                        float(mean_val - z_score * sem),
                        float(mean_val + z_score * sem)
                    ],
                    'percentiles': {
                        '5': float(np.percentile(values, 5)),
                        '25': float(np.percentile(values, 25)),
                        '50': float(np.percentile(values, 50)),
                        '75': float(np.percentile(values, 75)),
                        '95': float(np.percentile(values, 95))
                    }
                }
        
        return {
            'yield_rate': yield_rate,
            'successful_samples': successful_samples,
            'total_samples': n_samples,
            'metrics_statistics': statistics
        }
    
    def _genetic_algorithm_optimization(self, 
                                      objective_function: Callable,
                                      design_variables: Dict[str, Tuple[float, float]],
                                      constraints: List[Callable],
                                      max_evaluations: int) -> Dict[str, Any]:
        """Simplified genetic algorithm optimization."""
        
        # This is a simplified implementation
        # A full implementation would use a proper GA library
        
        population_size = min(50, max_evaluations // 20)
        n_generations = max_evaluations // population_size
        
        # Initialize population
        population = []
        var_names = list(design_variables.keys())
        var_bounds = list(design_variables.values())
        
        for _ in range(population_size):
            individual = {}
            for i, var_name in enumerate(var_names):
                low, high = var_bounds[i]
                individual[var_name] = np.random.uniform(low, high)
            population.append(individual)
        
        best_individual = None
        best_fitness = float('inf')
        
        for generation in range(n_generations):
            # Evaluate population
            fitness_scores = []
            for individual in population:
                try:
                    fitness = objective_function(individual)
                    
                    # Apply constraints
                    if constraints:
                        penalty = 0.0
                        for constraint in constraints:
                            violation = constraint(individual)
                            if violation > 0:
                                penalty += violation * 1000  # Penalty factor
                        fitness += penalty
                    
                    fitness_scores.append(fitness)
                    
                    if fitness < best_fitness:
                        best_fitness = fitness
                        best_individual = individual.copy()
                        
                except Exception as e:
                    fitness_scores.append(float('inf'))
            
            self.logger.debug(f"Generation {generation}: Best fitness = {best_fitness}")
            
            # Selection and evolution (simplified)
            # In practice, would implement proper selection, crossover, mutation
            
        return {
            'best_design': best_individual,
            'best_fitness': best_fitness,
            'generations': n_generations,
            'evaluations': n_generations * population_size
        }
    
    def _particle_swarm_optimization(self, 
                                   objective_function: Callable,
                                   design_variables: Dict[str, Tuple[float, float]],
                                   constraints: List[Callable],
                                   max_evaluations: int) -> Dict[str, Any]:
        """Simplified particle swarm optimization."""
        
        # Placeholder implementation
        # A full implementation would use a proper PSO library
        
        return {
            'best_design': {},
            'best_fitness': float('inf'),
            'iterations': 0,
            'evaluations': 0
        }
    
    def _update_dict_with_params(self, base_dict: Dict[str, Any], 
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Update dictionary with parameter values."""
        updated = base_dict.copy()
        updated.update(params)
        return updated
    
    def _generate_boundary_conditions(self, 
                                    physics_type: SolverType,
                                    params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate boundary conditions based on physics type and parameters."""
        
        if physics_type == SolverType.OPTICAL:
            return {
                'input_power': params.get('input_power', 1e-3),
                'wavelength': params.get('wavelength', 1550e-9)
            }
        elif physics_type == SolverType.THERMAL:
            return {
                'ambient_temperature': params.get('ambient_temp', 25) + 273.15,
                'heat_sources': []
            }
        elif physics_type == SolverType.ELECTRICAL:
            return {
                'voltage_sources': [],
                'current_sources': []
            }
        else:
            return {}
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'simulation_statistics': self.simulation_stats,
            'orchestrator_performance': self.orchestrator.optimizer.get_performance_report(),
            'memory_usage': self.optimizer._get_memory_usage(),
            'parallel_efficiency': self._calculate_parallel_efficiency()
        }
    
    def _calculate_parallel_efficiency(self) -> float:
        """Calculate parallel execution efficiency."""
        if self.simulation_stats['total_simulations'] == 0:
            return 0.0
        
        # Simplified efficiency calculation
        ideal_time = self.simulation_stats['total_compute_time'] / self.parallel_workers
        actual_time = self.simulation_stats['total_wall_time']
        
        if actual_time > 0:
            return min(1.0, ideal_time / actual_time)
        else:
            return 0.0
    
    def shutdown(self):
        """Shutdown the scalable simulator."""
        self.orchestrator.shutdown()
        self.logger.info("Scalable simulator shutdown completed")


# Factory functions
def create_scalable_simulator(config: Dict[str, Any]) -> ScalableMultiPhysicsSimulator:
    """Create scalable multi-physics simulator from configuration."""
    return ScalableMultiPhysicsSimulator(
        parallel_workers=config.get('parallel_workers'),
        use_gpu_acceleration=config.get('use_gpu_acceleration', True),
        memory_limit_gb=config.get('memory_limit_gb', 16.0),
        cache_enabled=config.get('cache_enabled', True)
    )

def benchmark_scalability(simulator: ScalableMultiPhysicsSimulator,
                         test_configs: List[Dict[str, Any]],
                         worker_counts: List[int] = None) -> Dict[str, Any]:
    """Benchmark simulator scalability across different worker counts."""
    
    worker_counts = worker_counts or [1, 2, 4, 8]
    results = {}
    
    for workers in worker_counts:
        # Create simulator with specific worker count
        test_simulator = ScalableMultiPhysicsSimulator(parallel_workers=workers)
        
        start_time = time.time()
        
        try:
            # Run test simulation
            test_results = test_simulator.simulate_parameter_sweep(
                base_geometry={'grid_size': (50, 50, 10)},
                base_materials={'silicon': {}},
                parameter_ranges={'wavelength': [1.5e-6, 1.55e-6]},
                max_concurrent=workers * 2
            )
            
            end_time = time.time()
            
            results[workers] = {
                'execution_time': end_time - start_time,
                'successful_simulations': sum(1 for r in test_results if r['success']),
                'total_simulations': len(test_results),
                'throughput': len(test_results) / (end_time - start_time)
            }
            
        except Exception as e:
            results[workers] = {
                'error': str(e)
            }
        finally:
            test_simulator.shutdown()
    
    return results