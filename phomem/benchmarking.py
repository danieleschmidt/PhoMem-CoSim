"""
Comprehensive benchmarking suite for PhoMem-CoSim performance evaluation.
"""

import logging
import time
import json
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import psutil
import matplotlib.pyplot as plt
import seaborn as sns

from .neural.networks import HybridNetwork
from .neural.training import HardwareAwareTrainer
from .simulator.multiphysics import MultiPhysicsSimulator
from .utils.performance import ProfileManager, MemoryMonitor
from .utils.logging import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    benchmark_name: str
    execution_time: float
    memory_usage: Dict[str, float]
    accuracy: Optional[float] = None
    throughput: Optional[float] = None
    energy_efficiency: Optional[float] = None
    scalability_metrics: Optional[Dict[str, Any]] = None
    system_info: Optional[Dict[str, Any]] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    suite_name: str
    total_time: float
    individual_results: List[BenchmarkResult]
    summary_metrics: Dict[str, Any]
    system_specifications: Dict[str, Any]
    timestamp: str


class PerformanceBenchmark:
    """Comprehensive performance benchmarking system."""
    
    def __init__(self, output_dir: Path = Path("./benchmarks")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(
            log_level=logging.INFO,
            log_file=self.output_dir / "benchmark.log"
        )
        
        # System monitoring
        self.memory_monitor = MemoryMonitor()
        
        # Benchmark registry
        self.benchmarks = {}
        self.register_default_benchmarks()
    
    def register_benchmark(self, name: str, benchmark_func: Callable):
        """Register a new benchmark function."""
        self.benchmarks[name] = benchmark_func
        logger.info(f"Registered benchmark: {name}")
    
    def register_default_benchmarks(self):
        """Register default benchmark suite."""
        self.register_benchmark("network_creation", self._benchmark_network_creation)
        self.register_benchmark("forward_pass", self._benchmark_forward_pass)
        self.register_benchmark("training_step", self._benchmark_training_step)
        self.register_benchmark("simulation_basic", self._benchmark_simulation_basic)
        self.register_benchmark("memory_scaling", self._benchmark_memory_scaling)
        self.register_benchmark("cpu_vs_gpu", self._benchmark_cpu_vs_gpu)
        self.register_benchmark("batch_processing", self._benchmark_batch_processing)
        self.register_benchmark("parameter_sweep", self._benchmark_parameter_sweep)
    
    def run_benchmark_suite(
        self,
        suite_name: str = "comprehensive",
        benchmarks_to_run: Optional[List[str]] = None,
        repetitions: int = 5
    ) -> BenchmarkSuite:
        """Run complete benchmark suite."""
        logger.info(f"Starting benchmark suite: {suite_name}")
        start_time = time.time()
        
        # Determine which benchmarks to run
        if benchmarks_to_run is None:
            benchmarks_to_run = list(self.benchmarks.keys())
        
        # Collect system information
        system_specs = self._collect_system_info()
        
        # Run individual benchmarks
        results = []
        
        for benchmark_name in benchmarks_to_run:
            if benchmark_name not in self.benchmarks:
                logger.warning(f"Unknown benchmark: {benchmark_name}")
                continue
            
            logger.info(f"Running benchmark: {benchmark_name}")
            
            # Run benchmark multiple times for statistical significance
            benchmark_results = []
            for rep in range(repetitions):
                try:
                    result = self._run_single_benchmark(
                        benchmark_name,
                        self.benchmarks[benchmark_name]
                    )
                    benchmark_results.append(result)
                except Exception as e:
                    logger.error(f"Benchmark {benchmark_name} failed (rep {rep}): {e}")
            
            # Aggregate results
            if benchmark_results:
                aggregated_result = self._aggregate_benchmark_results(
                    benchmark_name, benchmark_results
                )
                results.append(aggregated_result)
        
        # Calculate suite metrics
        total_time = time.time() - start_time
        summary_metrics = self._calculate_suite_summary(results)
        
        # Create suite result
        suite_result = BenchmarkSuite(
            suite_name=suite_name,
            total_time=total_time,
            individual_results=results,
            summary_metrics=summary_metrics,
            system_specifications=system_specs,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Save results
        self._save_benchmark_suite(suite_result)
        
        logger.info(f"Benchmark suite completed in {total_time:.2f}s")
        return suite_result
    
    def _run_single_benchmark(self, name: str, benchmark_func: Callable) -> BenchmarkResult:
        """Run a single benchmark with monitoring."""
        # Clear memory and trigger garbage collection
        import gc
        gc.collect()
        
        # Start monitoring
        start_memory = self.memory_monitor.get_memory_usage()
        start_time = time.time()
        
        with ProfileManager(enabled=True) as profiler:
            try:
                # Run benchmark
                result_data = benchmark_func()
                
                # Calculate metrics
                execution_time = time.time() - start_time
                end_memory = self.memory_monitor.get_memory_usage()
                memory_usage = {
                    'peak_memory_mb': end_memory.get('peak_memory_gb', 0) * 1024,
                    'memory_increase_mb': (end_memory.get('current_memory_gb', 0) - 
                                         start_memory.get('current_memory_gb', 0)) * 1024
                }
                
                # Create result
                result = BenchmarkResult(
                    benchmark_name=name,
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    system_info=self._collect_system_info(),
                    additional_metrics=result_data
                )
                
                # Add profiling data if available
                if hasattr(profiler, 'get_stats'):
                    result.additional_metrics['profiling'] = profiler.get_stats()
                
                return result
                
            except Exception as e:
                logger.error(f"Benchmark {name} execution failed: {e}")
                return BenchmarkResult(
                    benchmark_name=name,
                    execution_time=time.time() - start_time,
                    memory_usage={'error': str(e)},
                    additional_metrics={'error': str(e)}
                )
    
    def _aggregate_benchmark_results(
        self,
        benchmark_name: str,
        results: List[BenchmarkResult]
    ) -> BenchmarkResult:
        """Aggregate multiple benchmark runs."""
        if not results:
            raise ValueError("No results to aggregate")
        
        # Extract metrics
        execution_times = [r.execution_time for r in results if r.execution_time is not None]
        memory_usages = [r.memory_usage.get('peak_memory_mb', 0) for r in results]
        
        # Calculate statistics
        aggregated_metrics = {
            'num_repetitions': len(results),
            'execution_time_mean': np.mean(execution_times),
            'execution_time_std': np.std(execution_times),
            'execution_time_min': np.min(execution_times),
            'execution_time_max': np.max(execution_times),
            'memory_usage_mean': np.mean(memory_usages),
            'memory_usage_std': np.std(memory_usages),
            'success_rate': sum(1 for r in results if 'error' not in r.additional_metrics) / len(results)
        }
        
        # Aggregate additional metrics if present
        for result in results:
            for key, value in result.additional_metrics.items():
                if key != 'error' and isinstance(value, (int, float)):
                    metric_key = f"{key}_values"
                    if metric_key not in aggregated_metrics:
                        aggregated_metrics[metric_key] = []
                    aggregated_metrics[metric_key].append(value)
        
        # Calculate statistics for additional metrics
        for key in list(aggregated_metrics.keys()):
            if key.endswith('_values'):
                values = aggregated_metrics[key]
                base_key = key[:-7]  # Remove '_values'
                aggregated_metrics[f"{base_key}_mean"] = np.mean(values)
                aggregated_metrics[f"{base_key}_std"] = np.std(values)
                del aggregated_metrics[key]  # Remove raw values
        
        return BenchmarkResult(
            benchmark_name=benchmark_name,
            execution_time=aggregated_metrics['execution_time_mean'],
            memory_usage={
                'peak_memory_mb': aggregated_metrics['memory_usage_mean'],
                'memory_std_mb': aggregated_metrics['memory_usage_std']
            },
            system_info=results[0].system_info,
            additional_metrics=aggregated_metrics
        )
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for benchmarking context."""
        try:
            import platform
            
            # Basic system info
            system_info = {
                'platform': platform.system(),
                'platform_version': platform.version(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
            }
            
            # Memory info
            memory = psutil.virtual_memory()
            system_info['total_memory_gb'] = memory.total / (1024**3)
            system_info['available_memory_gb'] = memory.available / (1024**3)
            
            # CPU info
            system_info['cpu_count'] = psutil.cpu_count()
            system_info['cpu_count_physical'] = psutil.cpu_count(logical=False)
            system_info['cpu_freq_mhz'] = psutil.cpu_freq().current if psutil.cpu_freq() else None
            
            # JAX device info
            try:
                devices = jax.devices()
                system_info['jax_devices'] = [str(d) for d in devices]
                system_info['jax_default_backend'] = jax.default_backend()
            except:
                system_info['jax_devices'] = ['unavailable']
            
            # GPU info (if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    system_info['gpu_info'] = [
                        {
                            'name': gpu.name,
                            'memory_total_mb': gpu.memoryTotal,
                            'memory_free_mb': gpu.memoryFree
                        }
                        for gpu in gpus
                    ]
            except ImportError:
                system_info['gpu_info'] = 'GPUtil not available'
            
            return system_info
            
        except Exception as e:
            logger.warning(f"Failed to collect system info: {e}")
            return {'error': str(e)}
    
    def _calculate_suite_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate summary metrics for the benchmark suite."""
        if not results:
            return {}
        
        # Basic statistics
        total_execution_time = sum(r.execution_time for r in results)
        success_rate = sum(1 for r in results if 'error' not in r.additional_metrics) / len(results)
        
        # Memory statistics
        peak_memories = [r.memory_usage.get('peak_memory_mb', 0) for r in results]
        total_peak_memory = max(peak_memories) if peak_memories else 0
        
        # Performance ratings
        performance_scores = []
        for result in results:
            # Simple scoring based on execution time and success
            if 'error' in result.additional_metrics:
                score = 0.0
            else:
                # Normalize by expected ranges (these would be tuned based on experience)
                time_score = max(0, 10 - result.execution_time)  # 10s max expected
                memory_score = max(0, 10 - result.memory_usage.get('peak_memory_mb', 0) / 100)  # 1GB max expected
                score = (time_score + memory_score) / 2
            
            performance_scores.append(score)
        
        return {
            'total_benchmarks': len(results),
            'total_execution_time': total_execution_time,
            'success_rate': success_rate,
            'average_performance_score': np.mean(performance_scores),
            'total_peak_memory_mb': total_peak_memory,
            'benchmark_names': [r.benchmark_name for r in results]
        }
    
    def _save_benchmark_suite(self, suite: BenchmarkSuite):
        """Save benchmark suite results."""
        timestamp = suite.timestamp.replace(":", "-").replace(" ", "_")
        filename = f"benchmark_suite_{suite.suite_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Convert to JSON-serializable format
        suite_dict = {
            'suite_name': suite.suite_name,
            'total_time': suite.total_time,
            'timestamp': suite.timestamp,
            'system_specifications': suite.system_specifications,
            'summary_metrics': suite.summary_metrics,
            'individual_results': [
                {
                    'benchmark_name': r.benchmark_name,
                    'execution_time': r.execution_time,
                    'memory_usage': r.memory_usage,
                    'accuracy': r.accuracy,
                    'throughput': r.throughput,
                    'energy_efficiency': r.energy_efficiency,
                    'additional_metrics': r.additional_metrics
                }
                for r in suite.individual_results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(suite_dict, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {filepath}")
    
    # Individual benchmark implementations
    def _benchmark_network_creation(self) -> Dict[str, Any]:
        """Benchmark network creation performance."""
        network_sizes = [(4, [8, 4], 2), (16, [32, 16], 8), (64, [128, 64], 32)]
        creation_times = []
        
        for input_size, hidden_sizes, output_size in network_sizes:
            start_time = time.time()
            
            try:
                # Create photonic-memristive network
                network = HybridNetwork(
                    input_size=input_size,
                    hidden_sizes=hidden_sizes,
                    output_size=output_size,
                    photonic_layers=[0],
                    memristive_layers=[1, 2]
                )
                
                # Initialize parameters
                key = jax.random.PRNGKey(42)
                params = network.init(key, jnp.ones(input_size))
                
                creation_time = time.time() - start_time
                creation_times.append(creation_time)
                
            except Exception as e:
                logger.error(f"Network creation failed for size {input_size}: {e}")
                creation_times.append(float('inf'))
        
        return {
            'network_sizes': network_sizes,
            'creation_times': creation_times,
            'average_creation_time': np.mean([t for t in creation_times if t != float('inf')]),
            'scalability_factor': creation_times[-1] / creation_times[0] if creation_times[0] > 0 else 0
        }
    
    def _benchmark_forward_pass(self) -> Dict[str, Any]:
        """Benchmark forward pass performance."""
        batch_sizes = [1, 8, 32, 128]
        input_size = 16
        
        # Create test network
        network = HybridNetwork(
            input_size=input_size,
            hidden_sizes=[32, 16],
            output_size=8,
            photonic_layers=[0],
            memristive_layers=[1, 2]
        )
        
        # Initialize parameters
        key = jax.random.PRNGKey(42)
        params = network.init(key, jnp.ones(input_size))
        
        # Benchmark different batch sizes
        throughput_data = []
        
        for batch_size in batch_sizes:
            inputs = jax.random.normal(key, (batch_size, input_size))
            
            # Warm-up
            for _ in range(3):
                _ = network.apply(params, inputs)
            
            # Benchmark
            start_time = time.time()
            num_iterations = 100
            
            for _ in range(num_iterations):
                _ = network.apply(params, inputs)
            
            total_time = time.time() - start_time
            throughput = (batch_size * num_iterations) / total_time  # samples/second
            throughput_data.append(throughput)
        
        return {
            'batch_sizes': batch_sizes,
            'throughput_samples_per_second': throughput_data,
            'max_throughput': max(throughput_data),
            'throughput_scaling': throughput_data[-1] / throughput_data[0] if throughput_data[0] > 0 else 0
        }
    
    def _benchmark_training_step(self) -> Dict[str, Any]:
        """Benchmark training step performance."""
        network = HybridNetwork(
            input_size=8,
            hidden_sizes=[16, 8],
            output_size=4
        )
        
        # Create trainer
        trainer = HardwareAwareTrainer(
            learning_rate=1e-3,
            hardware_penalties={
                'optical_power': 0.1,
                'thermal': 0.01
            }
        )
        
        # Generate synthetic data
        key = jax.random.PRNGKey(42)
        batch_size = 32
        inputs = jax.random.normal(key, (batch_size, 8))
        targets = jax.random.normal(key, (batch_size, 4))
        
        # Initialize parameters
        params = network.init(key, jnp.ones(8))
        
        # Benchmark training steps
        training_times = []
        num_steps = 50
        
        for step in range(num_steps):
            start_time = time.time()
            
            # Single training step
            params, loss, metrics = trainer.training_step(
                params, inputs, targets, network
            )
            
            step_time = time.time() - start_time
            training_times.append(step_time)
        
        return {
            'num_training_steps': num_steps,
            'step_times': training_times,
            'average_step_time': np.mean(training_times),
            'steps_per_second': 1.0 / np.mean(training_times),
            'final_loss': float(loss) if 'loss' in locals() else None
        }
    
    def _benchmark_simulation_basic(self) -> Dict[str, Any]:
        """Benchmark basic multi-physics simulation."""
        # Create simple test network
        network = HybridNetwork(
            input_size=4,
            hidden_sizes=[8],
            output_size=2,
            photonic_layers=[0],
            memristive_layers=[1]
        )
        
        # Initialize simulator
        simulator = MultiPhysicsSimulator(
            optical_solver='BPM',
            thermal_solver='FEM',
            electrical_solver='SPICE',
            coupling='weak'
        )
        
        # Test different simulation durations
        durations = [0.1, 0.5, 1.0]  # seconds
        simulation_times = []
        
        inputs = jnp.array([1.0, 0.5, -0.5, 1.0])
        
        for duration in durations:
            start_time = time.time()
            
            try:
                results = simulator.simulate(
                    network=network,
                    inputs=inputs,
                    duration=duration,
                    save_fields=False
                )
                
                sim_time = time.time() - start_time
                simulation_times.append(sim_time)
                
            except Exception as e:
                logger.error(f"Simulation failed for duration {duration}: {e}")
                simulation_times.append(float('inf'))
        
        return {
            'simulation_durations': durations,
            'simulation_times': simulation_times,
            'average_simulation_time': np.mean([t for t in simulation_times if t != float('inf')]),
            'simulation_overhead': simulation_times[0] / durations[0] if durations[0] > 0 and simulation_times[0] != float('inf') else None
        }
    
    def _benchmark_memory_scaling(self) -> Dict[str, Any]:
        """Benchmark memory usage scaling with network size."""
        network_sizes = [
            (4, [8], 2),
            (8, [16, 8], 4),
            (16, [32, 16], 8),
            (32, [64, 32], 16)
        ]
        
        memory_usage = []
        parameter_counts = []
        
        for input_size, hidden_sizes, output_size in network_sizes:
            # Clear memory
            import gc
            gc.collect()
            
            # Measure initial memory
            initial_memory = self.memory_monitor.get_memory_usage()
            
            try:
                # Create network
                network = HybridNetwork(
                    input_size=input_size,
                    hidden_sizes=hidden_sizes,
                    output_size=output_size
                )
                
                # Initialize parameters
                key = jax.random.PRNGKey(42)
                params = network.init(key, jnp.ones(input_size))
                
                # Count parameters
                param_count = sum(p.size for p in jax.tree_leaves(params))
                parameter_counts.append(param_count)
                
                # Measure memory after creation
                final_memory = self.memory_monitor.get_memory_usage()
                memory_increase = (final_memory.get('current_memory_gb', 0) - 
                                 initial_memory.get('current_memory_gb', 0)) * 1024  # MB
                memory_usage.append(memory_increase)
                
            except Exception as e:
                logger.error(f"Memory benchmark failed for network size {input_size}: {e}")
                memory_usage.append(0)
                parameter_counts.append(0)
        
        return {
            'network_sizes': network_sizes,
            'memory_usage_mb': memory_usage,
            'parameter_counts': parameter_counts,
            'memory_per_parameter': [m/p if p > 0 else 0 for m, p in zip(memory_usage, parameter_counts)],
            'memory_scaling_factor': memory_usage[-1] / memory_usage[0] if memory_usage[0] > 0 else 0
        }
    
    def _benchmark_cpu_vs_gpu(self) -> Dict[str, Any]:
        """Benchmark CPU vs GPU performance."""
        network = HybridNetwork(
            input_size=16,
            hidden_sizes=[64, 32],
            output_size=8
        )
        
        # Test data
        key = jax.random.PRNGKey(42)
        inputs = jax.random.normal(key, (64, 16))  # Batch of 64
        
        results = {}
        
        # Test on different devices
        for device_type in ['cpu', 'gpu']:
            try:
                # Set JAX platform
                if device_type == 'gpu' and len(jax.devices('gpu')) == 0:
                    logger.info("GPU not available, skipping GPU benchmark")
                    continue
                
                with jax.default_device(jax.devices(device_type)[0]):
                    # Initialize parameters on device
                    params = network.init(key, jnp.ones(16))
                    
                    # Warm-up
                    for _ in range(5):
                        _ = network.apply(params, inputs)
                    
                    # Benchmark
                    start_time = time.time()
                    num_iterations = 100
                    
                    for _ in range(num_iterations):
                        _ = network.apply(params, inputs)
                    
                    total_time = time.time() - start_time
                    throughput = (64 * num_iterations) / total_time  # samples/second
                    
                    results[device_type] = {
                        'execution_time': total_time,
                        'throughput': throughput,
                        'time_per_sample': total_time / (64 * num_iterations)
                    }
                    
            except Exception as e:
                logger.error(f"Failed to benchmark {device_type}: {e}")
                results[device_type] = {'error': str(e)}
        
        # Calculate speedup if both succeeded
        if 'cpu' in results and 'gpu' in results and 'error' not in results['gpu']:
            cpu_time = results['cpu']['execution_time']
            gpu_time = results['gpu']['execution_time']
            results['gpu_speedup'] = cpu_time / gpu_time if gpu_time > 0 else 0
        
        return results
    
    def _benchmark_batch_processing(self) -> Dict[str, Any]:
        """Benchmark batch processing capabilities."""
        from .batch import BatchProcessor
        
        # Create batch processor
        batch_processor = BatchProcessor(
            output_dir=self.output_dir / "batch_test",
            max_workers=4
        )
        
        # Create simple jobs
        num_jobs = 10
        job_times = []
        
        start_time = time.time()
        
        try:
            # Add simulation jobs
            for i in range(num_jobs):
                network_config = {
                    'input_size': 4 + i % 4,
                    'hidden_sizes': [8 + i % 8],
                    'output_size': 2 + i % 2
                }
                
                simulation_config = {
                    'duration': 0.1,
                    'optical_solver': 'BPM'
                }
                
                inputs = np.random.random((1, network_config['input_size']))
                
                batch_processor.add_simulation_job(
                    job_id=f"test_job_{i}",
                    network_config=network_config,
                    simulation_config=simulation_config,
                    inputs=inputs
                )
            
            # Process batch
            batch_results = batch_processor.process_batch()
            
            total_batch_time = time.time() - start_time
            
            return {
                'num_jobs': num_jobs,
                'total_batch_time': total_batch_time,
                'completed_jobs': batch_results.completed_jobs,
                'failed_jobs': batch_results.failed_jobs,
                'success_rate': batch_results.completed_jobs / batch_results.total_jobs,
                'average_job_time': total_batch_time / num_jobs,
                'throughput_jobs_per_second': num_jobs / total_batch_time
            }
            
        except Exception as e:
            logger.error(f"Batch processing benchmark failed: {e}")
            return {'error': str(e)}
    
    def _benchmark_parameter_sweep(self) -> Dict[str, Any]:
        """Benchmark parameter sweep performance."""
        from .batch import BatchProcessor
        
        batch_processor = BatchProcessor(
            output_dir=self.output_dir / "sweep_test",
            max_workers=2
        )
        
        # Parameter sweep configuration
        base_config = {
            'input_size': 4,
            'hidden_sizes': [8],
            'output_size': 2
        }
        
        parameter_ranges = {
            'hidden_sizes': [[4], [8], [16]],
            'learning_rate': [1e-4, 1e-3, 1e-2]
        }
        
        start_time = time.time()
        
        try:
            # Create parameter sweep
            job_ids = batch_processor.add_parameter_sweep(
                base_job_id="sweep_test",
                job_type="training",
                base_config=base_config,
                parameter_ranges=parameter_ranges
            )
            
            # Process sweep (simplified - would normally take much longer)
            # For benchmarking, we'll just measure setup time
            setup_time = time.time() - start_time
            
            return {
                'num_parameter_combinations': len(job_ids),
                'parameter_ranges': parameter_ranges,
                'setup_time': setup_time,
                'jobs_per_second_setup': len(job_ids) / setup_time if setup_time > 0 else 0,
                'total_jobs_created': len(job_ids)
            }
            
        except Exception as e:
            logger.error(f"Parameter sweep benchmark failed: {e}")
            return {'error': str(e)}
    
    def generate_benchmark_report(self, suite: BenchmarkSuite) -> str:
        """Generate human-readable benchmark report."""
        report = []
        report.append(f"# PhoMem-CoSim Benchmark Report")
        report.append(f"**Suite**: {suite.suite_name}")
        report.append(f"**Date**: {suite.timestamp}")
        report.append(f"**Total Time**: {suite.total_time:.2f} seconds")
        report.append("")
        
        # System specifications
        report.append("## System Specifications")
        specs = suite.system_specifications
        report.append(f"- **Platform**: {specs.get('platform', 'Unknown')}")
        report.append(f"- **CPU**: {specs.get('processor', 'Unknown')} ({specs.get('cpu_count', '?')} cores)")
        report.append(f"- **Memory**: {specs.get('total_memory_gb', 0):.1f} GB")
        report.append(f"- **JAX Backend**: {specs.get('jax_default_backend', 'Unknown')}")
        report.append(f"- **JAX Devices**: {', '.join(specs.get('jax_devices', ['Unknown']))}")
        report.append("")
        
        # Summary metrics
        report.append("## Summary")
        summary = suite.summary_metrics
        report.append(f"- **Benchmarks Run**: {summary.get('total_benchmarks', 0)}")
        report.append(f"- **Success Rate**: {summary.get('success_rate', 0):.1%}")
        report.append(f"- **Average Performance Score**: {summary.get('average_performance_score', 0):.1f}/10")
        report.append(f"- **Peak Memory Usage**: {summary.get('total_peak_memory_mb', 0):.0f} MB")
        report.append("")
        
        # Individual results
        report.append("## Individual Benchmark Results")
        
        for result in suite.individual_results:
            report.append(f"### {result.benchmark_name}")
            report.append(f"- **Execution Time**: {result.execution_time:.3f}s")
            
            if result.memory_usage:
                peak_mem = result.memory_usage.get('peak_memory_mb', 0)
                report.append(f"- **Peak Memory**: {peak_mem:.1f} MB")
            
            # Add specific metrics based on benchmark type
            metrics = result.additional_metrics
            if 'throughput_samples_per_second' in metrics:
                max_throughput = max(metrics['throughput_samples_per_second'])
                report.append(f"- **Max Throughput**: {max_throughput:.0f} samples/sec")
            
            if 'success_rate' in metrics:
                report.append(f"- **Success Rate**: {metrics['success_rate']:.1%}")
            
            if 'average_creation_time' in metrics:
                report.append(f"- **Avg Creation Time**: {metrics['average_creation_time']:.3f}s")
            
            # Show errors if any
            if 'error' in metrics:
                report.append(f"- **Error**: {metrics['error']}")
            
            report.append("")
        
        return "\n".join(report)
    
    def plot_benchmark_results(
        self,
        suite: BenchmarkSuite,
        save_path: Optional[str] = None
    ):
        """Create visualization of benchmark results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'PhoMem-CoSim Benchmark Results: {suite.suite_name}', fontsize=16)
        
        # Extract data
        results = suite.individual_results
        names = [r.benchmark_name for r in results]
        times = [r.execution_time for r in results]
        memories = [r.memory_usage.get('peak_memory_mb', 0) for r in results]
        success_rates = [r.additional_metrics.get('success_rate', 1.0) for r in results]
        
        # Plot 1: Execution times
        ax = axes[0, 0]
        bars = ax.bar(range(len(names)), times, alpha=0.7)
        ax.set_xlabel('Benchmarks')
        ax.set_ylabel('Execution Time (s)')
        ax.set_title('Execution Time by Benchmark')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yscale('log')
        
        # Color bars by performance (green = fast, red = slow)
        max_time = max(times) if times else 1
        for i, bar in enumerate(bars):
            normalized_time = times[i] / max_time
            color = plt.cm.RdYlGn(1 - normalized_time)  # Invert so green is fast
            bar.set_color(color)
        
        # Plot 2: Memory usage
        ax = axes[0, 1]
        bars = ax.bar(range(len(names)), memories, alpha=0.7, color='skyblue')
        ax.set_xlabel('Benchmarks')
        ax.set_ylabel('Peak Memory (MB)')
        ax.set_title('Memory Usage by Benchmark')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        
        # Plot 3: Success rates
        ax = axes[1, 0]
        bars = ax.bar(range(len(names)), success_rates, alpha=0.7)
        ax.set_xlabel('Benchmarks')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate by Benchmark')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        
        # Color bars by success rate
        for i, bar in enumerate(bars):
            color = plt.cm.RdYlGn(success_rates[i])
            bar.set_color(color)
        
        # Plot 4: Performance radar chart
        ax = axes[1, 1]
        
        # Normalize metrics for radar chart
        norm_times = [1 - (t / max(times)) if max(times) > 0 else 0 for t in times]  # Invert so higher is better
        norm_memories = [1 - (m / max(memories)) if max(memories) > 0 else 0 for m in memories]  # Invert so higher is better
        
        # Create performance scores
        performance_scores = [(t + m + s) / 3 for t, m, s in zip(norm_times, norm_memories, success_rates)]
        
        bars = ax.bar(range(len(names)), performance_scores, alpha=0.7)
        ax.set_xlabel('Benchmarks')
        ax.set_ylabel('Performance Score')
        ax.set_title('Overall Performance Score\n(Higher is Better)')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        
        # Color bars by performance
        for i, bar in enumerate(bars):
            color = plt.cm.RdYlGn(performance_scores[i])
            bar.set_color(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Benchmark plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def run_comprehensive_benchmark(output_dir: str = "./benchmarks") -> BenchmarkSuite:
    """Run comprehensive benchmark suite and return results."""
    benchmarker = PerformanceBenchmark(Path(output_dir))
    
    # Run full benchmark suite
    suite = benchmarker.run_benchmark_suite(
        suite_name="comprehensive_v1.0",
        repetitions=3  # Reduce repetitions for faster execution
    )
    
    # Generate and save report
    report = benchmarker.generate_benchmark_report(suite)
    report_path = benchmarker.output_dir / f"benchmark_report_{suite.suite_name}.md"
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Generate plots
    plot_path = benchmarker.output_dir / f"benchmark_plots_{suite.suite_name}.png"
    benchmarker.plot_benchmark_results(suite, str(plot_path))
    
    logger.info(f"Comprehensive benchmark completed. Results saved to {benchmarker.output_dir}")
    
    return suite


if __name__ == "__main__":
    # Run benchmark when executed directly
    suite = run_comprehensive_benchmark()
    print(f"Benchmark suite completed with {suite.summary_metrics['success_rate']:.1%} success rate")