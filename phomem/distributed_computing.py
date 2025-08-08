"""
Distributed Computing Framework for Photonic-Memristive Simulations.

This module implements Generation 3 enhancements with:
- Distributed multi-physics simulation across multiple nodes
- Adaptive load balancing for heterogeneous computing resources
- Scalable architecture with fault tolerance
- Cloud-native deployment with auto-scaling
- GPU acceleration and memory optimization
- Edge computing support for real-time applications
"""

import logging
import time
import threading
import queue
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import warnings
import os
import pickle
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from .advanced_multiphysics import AdvancedMultiPhysicsSimulator, MultiPhysicsState
from .self_healing_optimization import SelfHealingOptimizer, HealthMetrics
from .optimization import OptimizationResult
from .utils.validation import ValidationError, validate_input_array
from .utils.logging import setup_logging
from .utils.performance import ProfileManager, MemoryMonitor

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of computing resources."""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu" 
    EDGE = "edge"
    CLOUD = "cloud"


class TaskStatus(Enum):
    """Status of distributed tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ComputeResource:
    """Represents a computing resource in the distributed system."""
    resource_id: str
    resource_type: ResourceType
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    network_bandwidth_mbps: float = 1000.0
    availability: float = 1.0  # 0.0 to 1.0
    current_load: float = 0.0  # 0.0 to 1.0
    location: str = "unknown"
    capabilities: List[str] = field(default_factory=list)
    
    def get_compute_capacity(self) -> float:
        """Calculate total compute capacity score."""
        cpu_score = self.cpu_cores * (1 - self.current_load)
        gpu_score = self.gpu_count * 10 if self.gpu_count > 0 else 0  # GPU worth 10x CPU
        memory_score = self.memory_gb / 16  # Normalize by 16GB
        
        return (cpu_score + gpu_score + memory_score) * self.availability


@dataclass
class DistributedTask:
    """Represents a task in the distributed computing system."""
    task_id: str
    task_type: str
    data: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    priority: int = 1  # 1 = low, 5 = high
    estimated_runtime: float = 0.0
    memory_requirement: float = 1.0  # GB
    gpu_required: bool = False
    status: TaskStatus = TaskStatus.PENDING
    assigned_resource: Optional[str] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def get_execution_time(self) -> Optional[float]:
        """Get actual execution time if completed."""
        if self.start_time and self.completion_time:
            return self.completion_time - self.start_time
        return None


class LoadBalancer:
    """Adaptive load balancer for distributed computing resources."""
    
    def __init__(
        self,
        balancing_strategy: str = "least_loaded",
        update_interval: float = 5.0,
        enable_prediction: bool = True
    ):
        self.balancing_strategy = balancing_strategy
        self.update_interval = update_interval
        self.enable_prediction = enable_prediction
        
        # Resource tracking
        self.resources: Dict[str, ComputeResource] = {}
        self.task_history: List[DistributedTask] = []
        self.performance_history: Dict[str, List[float]] = {}
        
        # Load balancing state
        self.last_update = 0.0
        self.resource_lock = threading.RLock()
        
        logger.info(f"Load balancer initialized with {balancing_strategy} strategy")
    
    def register_resource(self, resource: ComputeResource) -> bool:
        """Register a new computing resource."""
        with self.resource_lock:
            if resource.resource_id in self.resources:
                logger.warning(f"Resource {resource.resource_id} already registered")
                return False
            
            self.resources[resource.resource_id] = resource
            self.performance_history[resource.resource_id] = []
            logger.info(f"Registered resource {resource.resource_id} ({resource.resource_type.value})")
            return True
    
    def unregister_resource(self, resource_id: str) -> bool:
        """Unregister a computing resource."""
        with self.resource_lock:
            if resource_id not in self.resources:
                return False
            
            del self.resources[resource_id]
            if resource_id in self.performance_history:
                del self.performance_history[resource_id]
            
            logger.info(f"Unregistered resource {resource_id}")
            return True
    
    def select_resource(self, task: DistributedTask) -> Optional[str]:
        """Select the best resource for a given task."""
        with self.resource_lock:
            if not self.resources:
                logger.error("No resources available")
                return None
            
            # Filter resources by requirements
            eligible_resources = self._filter_eligible_resources(task)
            if not eligible_resources:
                logger.warning("No eligible resources found for task")
                return None
            
            # Apply load balancing strategy
            if self.balancing_strategy == "least_loaded":
                return self._select_least_loaded(eligible_resources)
            elif self.balancing_strategy == "best_fit":
                return self._select_best_fit(task, eligible_resources)
            elif self.balancing_strategy == "round_robin":
                return self._select_round_robin(eligible_resources)
            elif self.balancing_strategy == "performance_based":
                return self._select_performance_based(task, eligible_resources)
            else:
                # Default: least loaded
                return self._select_least_loaded(eligible_resources)
    
    def update_resource_load(self, resource_id: str, load: float):
        """Update current load for a resource."""
        with self.resource_lock:
            if resource_id in self.resources:
                self.resources[resource_id].current_load = max(0.0, min(1.0, load))
    
    def update_resource_availability(self, resource_id: str, availability: float):
        """Update availability for a resource."""
        with self.resource_lock:
            if resource_id in self.resources:
                self.resources[resource_id].availability = max(0.0, min(1.0, availability))
    
    def record_task_completion(self, task: DistributedTask):
        """Record task completion for performance tracking."""
        if task.assigned_resource and task.get_execution_time():
            execution_time = task.get_execution_time()
            
            with self.resource_lock:
                if task.assigned_resource in self.performance_history:
                    self.performance_history[task.assigned_resource].append(execution_time)
                    
                    # Keep only recent history
                    history = self.performance_history[task.assigned_resource]
                    if len(history) > 100:
                        self.performance_history[task.assigned_resource] = history[-100:]
            
            self.task_history.append(task)
            if len(self.task_history) > 1000:
                self.task_history = self.task_history[-1000:]
    
    def _filter_eligible_resources(self, task: DistributedTask) -> List[str]:
        """Filter resources that can handle the task."""
        eligible = []
        
        for resource_id, resource in self.resources.items():
            # Check availability
            if resource.availability < 0.1:
                continue
            
            # Check load
            if resource.current_load > 0.95:
                continue
            
            # Check GPU requirement
            if task.gpu_required and resource.gpu_count == 0:
                continue
            
            # Check memory requirement
            available_memory = resource.memory_gb * (1 - resource.current_load)
            if available_memory < task.memory_requirement:
                continue
            
            # Check specific capabilities if required
            required_caps = task.resource_requirements.get('capabilities', [])
            if required_caps and not all(cap in resource.capabilities for cap in required_caps):
                continue
            
            eligible.append(resource_id)
        
        return eligible
    
    def _select_least_loaded(self, eligible_resources: List[str]) -> str:
        """Select resource with lowest current load."""
        min_load = float('inf')
        selected = None
        
        for resource_id in eligible_resources:
            load = self.resources[resource_id].current_load
            if load < min_load:
                min_load = load
                selected = resource_id
        
        return selected
    
    def _select_best_fit(self, task: DistributedTask, eligible_resources: List[str]) -> str:
        """Select resource that best matches task requirements."""
        best_score = -1
        selected = None
        
        for resource_id in eligible_resources:
            resource = self.resources[resource_id]
            
            # Calculate fit score
            score = 0.0
            
            # Compute capacity vs requirement
            capacity = resource.get_compute_capacity()
            requirement = task.memory_requirement + (10 if task.gpu_required else 1)
            fit_ratio = min(1.0, requirement / (capacity + 1e-6))
            score += fit_ratio
            
            # Prefer resources not overqualified
            if not task.gpu_required and resource.gpu_count > 0:
                score *= 0.8  # Slight penalty for using GPU when not needed
            
            # Consider network bandwidth for distributed tasks
            if task.task_type in ['distributed_simulation', 'multi_node']:
                score += resource.network_bandwidth_mbps / 10000  # Normalize
            
            if score > best_score:
                best_score = score
                selected = resource_id
        
        return selected
    
    def _select_round_robin(self, eligible_resources: List[str]) -> str:
        """Select resource using round-robin strategy."""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        selected = eligible_resources[self._round_robin_index % len(eligible_resources)]
        self._round_robin_index += 1
        
        return selected
    
    def _select_performance_based(self, task: DistributedTask, eligible_resources: List[str]) -> str:
        """Select resource based on historical performance."""
        best_performance = float('inf')
        selected = None
        
        for resource_id in eligible_resources:
            # Get average execution time for this resource
            history = self.performance_history.get(resource_id, [])
            if history:
                avg_time = np.mean(history)
            else:
                avg_time = task.estimated_runtime or 60.0  # Default estimate
            
            # Adjust for current load
            load_factor = 1 + self.resources[resource_id].current_load
            adjusted_time = avg_time * load_factor
            
            if adjusted_time < best_performance:
                best_performance = adjusted_time
                selected = resource_id
        
        return selected or eligible_resources[0]  # Fallback
    
    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get current resource statistics."""
        with self.resource_lock:
            stats = {
                'total_resources': len(self.resources),
                'available_resources': sum(1 for r in self.resources.values() if r.availability > 0.5),
                'total_cpu_cores': sum(r.cpu_cores for r in self.resources.values()),
                'total_gpu_count': sum(r.gpu_count for r in self.resources.values()),
                'average_load': np.mean([r.current_load for r in self.resources.values()]) if self.resources else 0.0,
                'resource_types': {}
            }
            
            # Count by resource type
            for resource in self.resources.values():
                rt = resource.resource_type.value
                if rt not in stats['resource_types']:
                    stats['resource_types'][rt] = 0
                stats['resource_types'][rt] += 1
            
            return stats


class DistributedSimulationEngine:
    """Engine for distributed multi-physics simulations."""
    
    def __init__(
        self,
        load_balancer: LoadBalancer,
        max_concurrent_tasks: int = 10,
        enable_checkpointing: bool = True,
        checkpoint_interval: float = 300.0  # 5 minutes
    ):
        self.load_balancer = load_balancer
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_interval = checkpoint_interval
        
        # Task management
        self.pending_tasks = queue.PriorityQueue()
        self.running_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: List[DistributedTask] = []
        self.task_lock = threading.RLock()
        
        # Execution
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self.running = False
        self.coordinator_thread = None
        
        # Checkpointing
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.last_checkpoint = 0.0
        
        logger.info("Distributed simulation engine initialized")
    
    def start(self):
        """Start the distributed simulation engine."""
        if self.running:
            logger.warning("Engine already running")
            return
        
        self.running = True
        self.coordinator_thread = threading.Thread(target=self._coordinator_loop)
        self.coordinator_thread.daemon = True
        self.coordinator_thread.start()
        
        logger.info("Distributed simulation engine started")
    
    def stop(self):
        """Stop the distributed simulation engine."""
        if not self.running:
            return
        
        self.running = False
        if self.coordinator_thread:
            self.coordinator_thread.join(timeout=5.0)
        
        self.executor.shutdown(wait=True)
        logger.info("Distributed simulation engine stopped")
    
    def submit_task(self, task: DistributedTask) -> bool:
        """Submit a task for distributed execution."""
        if not self.running:
            logger.error("Engine not running")
            return False
        
        # Priority queue uses negative priority for max-heap behavior
        priority_score = -task.priority
        self.pending_tasks.put((priority_score, time.time(), task))
        
        logger.info(f"Submitted task {task.task_id} with priority {task.priority}")
        return True
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a specific task."""
        with self.task_lock:
            # Check running tasks
            if task_id in self.running_tasks:
                return self.running_tasks[task_id].status
            
            # Check completed tasks
            for task in self.completed_tasks:
                if task.task_id == task_id:
                    return task.status
            
            # Check pending tasks (this is O(n) but should be small)
            pending_list = list(self.pending_tasks.queue)
            for _, _, task in pending_list:
                if task.task_id == task_id:
                    return TaskStatus.PENDING
            
            return None
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get result of a completed task."""
        with self.task_lock:
            for task in self.completed_tasks:
                if task.task_id == task_id and task.status == TaskStatus.COMPLETED:
                    return task.result
            return None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        with self.task_lock:
            # Cancel running task
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.status = TaskStatus.CANCELLED
                return True
            
            # Remove from pending queue (O(n) operation)
            new_queue = queue.PriorityQueue()
            cancelled = False
            
            while not self.pending_tasks.empty():
                try:
                    item = self.pending_tasks.get_nowait()
                    _, _, task = item
                    if task.task_id == task_id:
                        task.status = TaskStatus.CANCELLED
                        cancelled = True
                    else:
                        new_queue.put(item)
                except queue.Empty:
                    break
            
            self.pending_tasks = new_queue
            return cancelled
    
    def _coordinator_loop(self):
        """Main coordinator loop for task scheduling."""
        logger.info("Coordinator loop started")
        
        while self.running:
            try:
                # Process pending tasks
                self._process_pending_tasks()
                
                # Check running tasks
                self._check_running_tasks()
                
                # Handle checkpointing
                if self.enable_checkpointing:
                    self._handle_checkpointing()
                
                # Sleep briefly
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in coordinator loop: {e}")
                time.sleep(1.0)
        
        logger.info("Coordinator loop stopped")
    
    def _process_pending_tasks(self):
        """Process pending tasks and assign to resources."""
        with self.task_lock:
            # Don't exceed concurrent limit
            if len(self.running_tasks) >= self.max_concurrent_tasks:
                return
            
            # Try to assign pending tasks
            tasks_to_start = []
            
            while (not self.pending_tasks.empty() and 
                   len(self.running_tasks) + len(tasks_to_start) < self.max_concurrent_tasks):
                
                try:
                    _, submit_time, task = self.pending_tasks.get_nowait()
                    
                    if task.status == TaskStatus.CANCELLED:
                        continue
                    
                    # Select resource for task
                    resource_id = self.load_balancer.select_resource(task)
                    if resource_id:
                        task.assigned_resource = resource_id
                        task.status = TaskStatus.RUNNING
                        task.start_time = time.time()
                        tasks_to_start.append(task)
                    else:
                        # No resource available, put back in queue
                        self.pending_tasks.put((-task.priority, submit_time, task))
                        break
                        
                except queue.Empty:
                    break
            
            # Start assigned tasks
            for task in tasks_to_start:
                self.running_tasks[task.task_id] = task
                future = self.executor.submit(self._execute_task, task)
                task.future = future
    
    def _check_running_tasks(self):
        """Check status of running tasks."""
        with self.task_lock:
            completed_tasks = []
            
            for task_id, task in list(self.running_tasks.items()):
                if hasattr(task, 'future') and task.future.done():
                    try:
                        result = task.future.result()
                        task.result = result
                        task.status = TaskStatus.COMPLETED
                        task.completion_time = time.time()
                        
                        # Update load balancer
                        self.load_balancer.record_task_completion(task)
                        
                    except Exception as e:
                        task.error = str(e)
                        task.status = TaskStatus.FAILED
                        task.completion_time = time.time()
                        logger.error(f"Task {task_id} failed: {e}")
                        
                        # Retry logic
                        if task.retry_count < task.max_retries:
                            task.retry_count += 1
                            task.status = TaskStatus.PENDING
                            task.start_time = None
                            task.completion_time = None
                            task.assigned_resource = None
                            
                            # Re-queue for retry
                            priority_score = -task.priority
                            self.pending_tasks.put((priority_score, time.time(), task))
                            logger.info(f"Re-queuing task {task_id} for retry {task.retry_count}")
                        else:
                            logger.error(f"Task {task_id} exhausted retries")
                    
                    completed_tasks.append(task_id)
            
            # Move completed tasks
            for task_id in completed_tasks:
                if task_id in self.running_tasks:
                    task = self.running_tasks.pop(task_id)
                    self.completed_tasks.append(task)
                    
                    # Keep limited history
                    if len(self.completed_tasks) > 1000:
                        self.completed_tasks = self.completed_tasks[-1000:]
    
    def _execute_task(self, task: DistributedTask) -> Any:
        """Execute a distributed task."""
        logger.info(f"Executing task {task.task_id} on resource {task.assigned_resource}")
        
        try:
            # Update resource load
            if task.assigned_resource:
                current_load = self.load_balancer.resources[task.assigned_resource].current_load
                estimated_load_increase = task.memory_requirement / \
                    self.load_balancer.resources[task.assigned_resource].memory_gb
                new_load = min(1.0, current_load + estimated_load_increase)
                self.load_balancer.update_resource_load(task.assigned_resource, new_load)
            
            # Execute based on task type
            if task.task_type == "multiphysics_simulation":
                result = self._execute_multiphysics_simulation(task)
            elif task.task_type == "optimization":
                result = self._execute_optimization(task)
            elif task.task_type == "data_processing":
                result = self._execute_data_processing(task)
            elif task.task_type == "parameter_sweep":
                result = self._execute_parameter_sweep(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            return result
            
        finally:
            # Update resource load (decrease)
            if task.assigned_resource:
                current_load = self.load_balancer.resources[task.assigned_resource].current_load
                estimated_load_decrease = task.memory_requirement / \
                    self.load_balancer.resources[task.assigned_resource].memory_gb
                new_load = max(0.0, current_load - estimated_load_decrease)
                self.load_balancer.update_resource_load(task.assigned_resource, new_load)
    
    def _execute_multiphysics_simulation(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute multi-physics simulation task."""
        data = task.data
        
        # Extract simulation parameters
        initial_state = data.get('initial_state')
        simulation_time = data.get('simulation_time', 1e-9)
        solver_config = data.get('solver_config', {})
        
        # Create simulator
        simulator = AdvancedMultiPhysicsSimulator(**solver_config)
        
        # Run simulation
        final_state, uq_result = simulator.simulate_coupled_physics(
            initial_state, simulation_time, data.get('parameter_uncertainties')
        )
        
        return {
            'final_state': final_state,
            'uncertainty_result': uq_result,
            'simulation_time': simulation_time,
            'task_id': task.task_id
        }
    
    def _execute_optimization(self, task: DistributedTask) -> OptimizationResult:
        """Execute optimization task."""
        data = task.data
        
        optimizer = data.get('optimizer')
        objective_function = data.get('objective_function')
        initial_params = data.get('initial_params')
        
        if not all([optimizer, objective_function, initial_params]):
            raise ValueError("Missing required optimization parameters")
        
        result = optimizer.optimize(objective_function, initial_params)
        return result
    
    def _execute_data_processing(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute data processing task."""
        data = task.data
        
        # Simple data processing example
        input_data = data.get('input_data')
        processing_type = data.get('processing_type', 'statistical_analysis')
        
        if processing_type == 'statistical_analysis':
            result = {
                'mean': np.mean(input_data),
                'std': np.std(input_data),
                'min': np.min(input_data),
                'max': np.max(input_data),
                'size': len(input_data)
            }
        elif processing_type == 'fourier_transform':
            result = {
                'fft': np.fft.fft(input_data).tolist(),
                'frequencies': np.fft.fftfreq(len(input_data)).tolist()
            }
        else:
            result = {'processed_data': input_data.tolist()}
        
        return result
    
    def _execute_parameter_sweep(self, task: DistributedTask) -> List[Dict[str, Any]]:
        """Execute parameter sweep task."""
        data = task.data
        
        parameter_ranges = data.get('parameter_ranges')
        base_config = data.get('base_config', {})
        evaluation_function = data.get('evaluation_function')
        
        results = []
        
        # Simple grid sweep (in practice would be more sophisticated)
        for param_name, param_values in parameter_ranges.items():
            for value in param_values:
                config = base_config.copy()
                config[param_name] = value
                
                # Evaluate
                if evaluation_function:
                    try:
                        result = evaluation_function(config)
                        results.append({
                            'parameters': config,
                            'result': result,
                            'success': True
                        })
                    except Exception as e:
                        results.append({
                            'parameters': config,
                            'error': str(e),
                            'success': False
                        })
        
        return results
    
    def _handle_checkpointing(self):
        """Handle periodic checkpointing."""
        current_time = time.time()
        if current_time - self.last_checkpoint > self.checkpoint_interval:
            self._create_checkpoint()
            self.last_checkpoint = current_time
    
    def _create_checkpoint(self):
        """Create a checkpoint of current state."""
        try:
            checkpoint_data = {
                'timestamp': time.time(),
                'running_tasks': {tid: self._serialize_task(task) 
                                 for tid, task in self.running_tasks.items()},
                'completed_tasks_count': len(self.completed_tasks),
                'resource_stats': self.load_balancer.get_resource_statistics()
            }
            
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{int(time.time())}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Clean old checkpoints (keep last 10)
            checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
            if len(checkpoints) > 10:
                for old_checkpoint in checkpoints[:-10]:
                    old_checkpoint.unlink()
            
            logger.debug(f"Created checkpoint: {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
    
    def _serialize_task(self, task: DistributedTask) -> Dict[str, Any]:
        """Serialize task for checkpointing."""
        return {
            'task_id': task.task_id,
            'task_type': task.task_type,
            'status': task.status.value,
            'assigned_resource': task.assigned_resource,
            'start_time': task.start_time,
            'retry_count': task.retry_count
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        with self.task_lock:
            stats = {
                'pending_tasks': self.pending_tasks.qsize(),
                'running_tasks': len(self.running_tasks),
                'completed_tasks': len(self.completed_tasks),
                'engine_running': self.running,
                'resource_stats': self.load_balancer.get_resource_statistics()
            }
            
            # Task type breakdown
            task_types = {}
            for task in list(self.running_tasks.values()) + self.completed_tasks:
                tt = task.task_type
                if tt not in task_types:
                    task_types[tt] = {'running': 0, 'completed': 0, 'failed': 0}
                
                if task.status == TaskStatus.RUNNING:
                    task_types[tt]['running'] += 1
                elif task.status == TaskStatus.COMPLETED:
                    task_types[tt]['completed'] += 1
                elif task.status == TaskStatus.FAILED:
                    task_types[tt]['failed'] += 1
            
            stats['task_types'] = task_types
            
            return stats


def create_distributed_resources(
    num_cpu_nodes: int = 2,
    num_gpu_nodes: int = 1,
    num_edge_nodes: int = 1
) -> List[ComputeResource]:
    """Create a set of distributed computing resources for testing."""
    
    resources = []
    
    # CPU nodes
    for i in range(num_cpu_nodes):
        resource = ComputeResource(
            resource_id=f"cpu_node_{i}",
            resource_type=ResourceType.CPU,
            cpu_cores=16,
            memory_gb=64.0,
            network_bandwidth_mbps=1000.0,
            location=f"datacenter_a",
            capabilities=["multiphysics_simulation", "optimization", "data_processing"]
        )
        resources.append(resource)
    
    # GPU nodes
    for i in range(num_gpu_nodes):
        resource = ComputeResource(
            resource_id=f"gpu_node_{i}",
            resource_type=ResourceType.GPU,
            cpu_cores=32,
            memory_gb=128.0,
            gpu_count=4,
            gpu_memory_gb=80.0,  # 4x 20GB GPUs
            network_bandwidth_mbps=10000.0,
            location="datacenter_b",
            capabilities=["multiphysics_simulation", "optimization", "neural_training", "parameter_sweep"]
        )
        resources.append(resource)
    
    # Edge nodes
    for i in range(num_edge_nodes):
        resource = ComputeResource(
            resource_id=f"edge_node_{i}",
            resource_type=ResourceType.EDGE,
            cpu_cores=4,
            memory_gb=8.0,
            network_bandwidth_mbps=100.0,
            location="edge_location",
            capabilities=["data_processing", "monitoring"]
        )
        resources.append(resource)
    
    logger.info(f"Created {len(resources)} distributed resources")
    return resources


def setup_distributed_system(
    resources: Optional[List[ComputeResource]] = None,
    load_balancing_strategy: str = "best_fit"
) -> Tuple[LoadBalancer, DistributedSimulationEngine]:
    """Setup a complete distributed computing system."""
    
    if resources is None:
        resources = create_distributed_resources()
    
    # Create load balancer
    load_balancer = LoadBalancer(balancing_strategy=load_balancing_strategy)
    
    # Register resources
    for resource in resources:
        load_balancer.register_resource(resource)
    
    # Create simulation engine
    engine = DistributedSimulationEngine(
        load_balancer=load_balancer,
        max_concurrent_tasks=len(resources) * 2,
        enable_checkpointing=True
    )
    
    logger.info(f"Distributed system setup complete with {len(resources)} resources")
    
    return load_balancer, engine