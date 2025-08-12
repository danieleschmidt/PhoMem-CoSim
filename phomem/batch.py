"""
Batch processing capabilities for large-scale PhoMem-CoSim operations.
"""

import logging
import json
import pickle
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import time

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from .config import PhoMemConfig
from .neural.networks import HybridNetwork
# from .neural.training import HardwareAwareTrainer  # TODO: Implement HardwareAwareTrainer
from .simulator.core import MultiPhysicsSimulator
from .utils.logging import setup_logging
from .utils.performance import PerformanceOptimizer, MemoryManager

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """Represents a single batch job."""
    job_id: str
    job_type: str  # 'simulation', 'training', 'analysis'
    config: Dict[str, Any]
    inputs: Optional[Dict[str, Any]] = None
    priority: int = 0  # Higher priority jobs run first
    dependencies: List[str] = field(default_factory=list)
    max_retries: int = 3
    timeout: Optional[float] = None  # seconds
    
    # Status tracking
    status: str = 'pending'  # 'pending', 'running', 'completed', 'failed', 'cancelled'
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None
    retries: int = 0
    results_path: Optional[str] = None


@dataclass
class BatchResults:
    """Results from batch processing."""
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    cancelled_jobs: int
    total_time: float
    job_results: Dict[str, Dict[str, Any]]
    performance_metrics: Dict[str, Any]


class BatchProcessor:
    """High-performance batch processing engine."""
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        max_workers: Optional[int] = None,
        use_gpu: bool = True,
        memory_limit: Optional[float] = None,  # GB
        enable_profiling: bool = False
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Worker configuration
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.use_gpu = use_gpu and jax.devices('gpu')
        self.memory_limit = memory_limit
        
        # Setup logging and profiling
        setup_logging(
            log_level=logging.INFO,
            log_file=self.output_dir / "batch_processing.log",
            console_output=True
        )
        
        self.profiling_enabled = enable_profiling
        self.memory_monitor = MemoryMonitor(limit_gb=memory_limit)
        
        # Job management
        self.jobs: Dict[str, BatchJob] = {}
        self.job_queue: List[str] = []
        self.completed_jobs: List[str] = []
        self.failed_jobs: List[str] = []
        
        logger.info(f"Initialized BatchProcessor with {self.max_workers} workers")
        if self.use_gpu:
            logger.info(f"GPU acceleration enabled: {len(jax.devices('gpu'))} devices")
    
    def add_job(self, job: BatchJob) -> str:
        """Add a job to the processing queue."""
        if job.job_id in self.jobs:
            raise ValueError(f"Job ID {job.job_id} already exists")
        
        self.jobs[job.job_id] = job
        self.job_queue.append(job.job_id)
        
        logger.info(f"Added job {job.job_id} (type: {job.job_type})")
        return job.job_id
    
    def add_simulation_job(
        self,
        job_id: str,
        network_config: Dict[str, Any],
        simulation_config: Dict[str, Any],
        inputs: np.ndarray,
        **kwargs
    ) -> str:
        """Add a simulation job."""
        job = BatchJob(
            job_id=job_id,
            job_type='simulation',
            config={
                'network': network_config,
                'simulation': simulation_config
            },
            inputs={'data': inputs},
            **kwargs
        )
        return self.add_job(job)
    
    def add_training_job(
        self,
        job_id: str,
        network_config: Dict[str, Any],
        training_config: Dict[str, Any],
        training_data: np.ndarray,
        **kwargs
    ) -> str:
        """Add a training job."""
        job = BatchJob(
            job_id=job_id,
            job_type='training',
            config={
                'network': network_config,
                'training': training_config
            },
            inputs={'training_data': training_data},
            **kwargs
        )
        return self.add_job(job)
    
    def add_parameter_sweep(
        self,
        base_job_id: str,
        job_type: str,
        base_config: Dict[str, Any],
        parameter_ranges: Dict[str, List[Any]],
        base_inputs: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Add parameter sweep jobs."""
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        job_ids = []
        for i, combination in enumerate(product(*param_values)):
            # Create modified config
            sweep_config = base_config.copy()
            for param_name, value in zip(param_names, combination):
                # Handle nested parameter paths like 'training.learning_rate'
                if '.' in param_name:
                    section, key = param_name.split('.', 1)
                    if section not in sweep_config:
                        sweep_config[section] = {}
                    sweep_config[section][key] = value
                else:
                    sweep_config[param_name] = value
            
            # Create job
            job_id = f"{base_job_id}_sweep_{i:04d}"
            job = BatchJob(
                job_id=job_id,
                job_type=job_type,
                config=sweep_config,
                inputs=base_inputs.copy() if base_inputs else None
            )
            
            job_ids.append(self.add_job(job))
        
        logger.info(f"Added parameter sweep: {len(job_ids)} jobs")
        return job_ids
    
    def add_monte_carlo_jobs(
        self,
        base_job_id: str,
        job_type: str,
        base_config: Dict[str, Any],
        variability_config: Dict[str, Dict[str, float]],
        n_samples: int,
        base_inputs: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Add Monte Carlo variability analysis jobs."""
        job_ids = []
        
        for i in range(n_samples):
            # Generate random variations
            mc_config = base_config.copy()
            
            for section, variations in variability_config.items():
                if section not in mc_config:
                    mc_config[section] = {}
                
                for param, std_dev in variations.items():
                    # Add Gaussian noise to parameter
                    base_value = mc_config[section].get(param, 0)
                    varied_value = base_value + np.random.normal(0, std_dev)
                    mc_config[section][param] = varied_value
            
            # Create job
            job_id = f"{base_job_id}_mc_{i:04d}"
            job = BatchJob(
                job_id=job_id,
                job_type=job_type,
                config=mc_config,
                inputs=base_inputs.copy() if base_inputs else None
            )
            
            job_ids.append(self.add_job(job))
        
        logger.info(f"Added Monte Carlo analysis: {len(job_ids)} jobs")
        return job_ids
    
    def get_ready_jobs(self) -> List[str]:
        """Get jobs that are ready to run (dependencies satisfied)."""
        ready_jobs = []
        
        for job_id in self.job_queue:
            job = self.jobs[job_id]
            
            if job.status != 'pending':
                continue
            
            # Check dependencies
            deps_satisfied = all(
                self.jobs[dep_id].status == 'completed'
                for dep_id in job.dependencies
                if dep_id in self.jobs
            )
            
            if deps_satisfied:
                ready_jobs.append(job_id)
        
        # Sort by priority (higher first)
        ready_jobs.sort(key=lambda jid: self.jobs[jid].priority, reverse=True)
        
        return ready_jobs
    
    def _execute_job(self, job_id: str) -> Dict[str, Any]:
        """Execute a single job (runs in worker process)."""
        job = self.jobs[job_id]
        
        try:
            # Update job status
            job.status = 'running'
            job.start_time = time.time()
            
            # Execute based on job type
            if job.job_type == 'simulation':
                results = self._run_simulation(job)
            elif job.job_type == 'training':
                results = self._run_training(job)
            elif job.job_type == 'analysis':
                results = self._run_analysis(job)
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")
            
            # Save results
            results_file = self.output_dir / f"{job_id}_results.pkl"
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)
            
            job.results_path = str(results_file)
            job.status = 'completed'
            job.end_time = time.time()
            
            logger.info(f"Job {job_id} completed in {job.end_time - job.start_time:.2f}s")
            
            return {
                'job_id': job_id,
                'status': 'completed',
                'results_path': job.results_path,
                'execution_time': job.end_time - job.start_time
            }
            
        except Exception as e:
            job.status = 'failed'
            job.end_time = time.time()
            job.error = str(e)
            job.retries += 1
            
            logger.error(f"Job {job_id} failed: {e}")
            
            return {
                'job_id': job_id,
                'status': 'failed',
                'error': str(e),
                'retries': job.retries
            }
    
    def _run_simulation(self, job: BatchJob) -> Dict[str, Any]:
        """Run simulation job."""
        # Initialize simulator
        sim_config = job.config['simulation']
        simulator = MultiPhysicsSimulator(
            optical_solver=sim_config.get('optical_solver', 'BPM'),
            thermal_solver=sim_config.get('thermal_solver', 'FEM'),
            electrical_solver=sim_config.get('electrical_solver', 'SPICE'),
            coupling=sim_config.get('coupling', 'weak')
        )
        
        # Create network
        network = HybridNetwork.from_config(job.config['network'])
        
        # Run simulation
        inputs = jnp.array(job.inputs['data'])
        results = simulator.simulate(
            network=network,
            inputs=inputs,
            duration=sim_config.get('duration', 1.0),
            save_fields=sim_config.get('save_fields', False)
        )
        
        return {
            'simulation_results': results,
            'network_config': job.config['network'],
            'simulation_config': sim_config
        }
    
    def _run_training(self, job: BatchJob) -> Dict[str, Any]:
        """Run training job."""
        # Initialize trainer
        train_config = job.config['training']
        trainer = HardwareAwareTrainer(
            learning_rate=train_config.get('learning_rate', 1e-3),
            hardware_penalties=train_config.get('hardware_penalties', {}),
            optimization_config=train_config.get('optimization', {})
        )
        
        # Create network
        network = HybridNetwork.from_config(job.config['network'])
        
        # Run training
        training_data = job.inputs['training_data']
        trained_network, history = trainer.train(
            network=network,
            data=training_data,
            epochs=train_config.get('epochs', 100),
            batch_size=train_config.get('batch_size', 32),
            validation_split=train_config.get('validation_split', 0.2)
        )
        
        return {
            'trained_network': trained_network,
            'training_history': history,
            'network_config': job.config['network'],
            'training_config': train_config
        }
    
    def _run_analysis(self, job: BatchJob) -> Dict[str, Any]:
        """Run analysis job."""
        # Import analysis modules
        from .simulator.optimization import PerformanceAnalyzer, VariabilityAnalyzer
        
        # Initialize analyzers based on config
        analysis_config = job.config.get('analysis', {})
        results = {}
        
        if 'performance' in analysis_config.get('analyses', []):
            analyzer = PerformanceAnalyzer()
            # Load subject for analysis
            if 'network_path' in job.inputs:
                subject = HybridNetwork.load(job.inputs['network_path'])
            else:
                subject = HybridNetwork.from_config(job.config['network'])
            
            results['performance'] = analyzer.analyze(subject)
        
        if 'variability' in analysis_config.get('analyses', []):
            analyzer = VariabilityAnalyzer()
            results['variability'] = analyzer.analyze(
                subject,
                n_samples=analysis_config.get('variability_samples', 1000)
            )
        
        return {
            'analysis_results': results,
            'analysis_config': analysis_config
        }
    
    def process_batch(
        self,
        max_concurrent: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> BatchResults:
        """Process all jobs in the batch."""
        start_time = time.time()
        max_concurrent = max_concurrent or self.max_workers
        
        logger.info(f"Starting batch processing: {len(self.jobs)} total jobs")
        
        # Setup progress tracking
        pbar = tqdm(total=len(self.jobs), desc="Processing jobs")
        
        with ProfileManager(enabled=self.profiling_enabled, output_dir=self.output_dir):
            if self.use_gpu:
                # Use thread pool for GPU jobs (JAX handles parallelism)
                with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
                    results = self._process_with_executor(executor, pbar, progress_callback)
            else:
                # Use process pool for CPU jobs
                with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
                    results = self._process_with_executor(executor, pbar, progress_callback)
        
        pbar.close()
        
        # Compile results
        total_time = time.time() - start_time
        batch_results = BatchResults(
            total_jobs=len(self.jobs),
            completed_jobs=len(self.completed_jobs),
            failed_jobs=len(self.failed_jobs),
            cancelled_jobs=len([j for j in self.jobs.values() if j.status == 'cancelled']),
            total_time=total_time,
            job_results=results,
            performance_metrics=self.memory_monitor.get_stats()
        )
        
        # Save batch summary
        summary_path = self.output_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            # Convert to JSON-serializable format
            summary = {
                'total_jobs': batch_results.total_jobs,
                'completed_jobs': batch_results.completed_jobs,
                'failed_jobs': batch_results.failed_jobs,
                'cancelled_jobs': batch_results.cancelled_jobs,
                'total_time': batch_results.total_time,
                'throughput': batch_results.completed_jobs / batch_results.total_time,
                'success_rate': batch_results.completed_jobs / batch_results.total_jobs,
                'performance_metrics': batch_results.performance_metrics
            }
            json.dump(summary, f, indent=2)
        
        logger.info(f"Batch processing completed in {total_time:.2f}s")
        logger.info(f"Success rate: {batch_results.completed_jobs}/{batch_results.total_jobs}")
        
        return batch_results
    
    def _process_with_executor(self, executor, pbar, progress_callback):
        """Process jobs with given executor."""
        futures = {}
        results = {}
        
        while self.job_queue or futures:
            # Submit ready jobs
            ready_jobs = self.get_ready_jobs()
            for job_id in ready_jobs[:self.max_workers - len(futures)]:
                if job_id not in futures:
                    future = executor.submit(self._execute_job, job_id)
                    futures[future] = job_id
                    self.job_queue.remove(job_id)
            
            # Process completed jobs
            if futures:
                for future in as_completed(futures, timeout=1.0):
                    job_id = futures.pop(future)
                    result = future.result()
                    
                    results[job_id] = result
                    
                    if result['status'] == 'completed':
                        self.completed_jobs.append(job_id)
                    elif result['status'] == 'failed':
                        # Retry if possible
                        job = self.jobs[job_id]
                        if job.retries < job.max_retries:
                            job.status = 'pending'
                            self.job_queue.append(job_id)
                            logger.info(f"Retrying job {job_id} ({job.retries}/{job.max_retries})")
                        else:
                            self.failed_jobs.append(job_id)
                    
                    pbar.update(1)
                    
                    if progress_callback:
                        progress_callback(result)
        
        return results


class BatchAnalyzer:
    """Analyze results from batch processing."""
    
    def __init__(self, batch_dir: Union[str, Path]):
        self.batch_dir = Path(batch_dir)
        self.results = {}
        self.summary = {}
        
        # Load batch summary
        summary_path = self.batch_dir / "batch_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                self.summary = json.load(f)
    
    def load_results(self) -> Dict[str, Any]:
        """Load all job results."""
        result_files = list(self.batch_dir.glob("*_results.pkl"))
        
        for result_file in result_files:
            job_id = result_file.stem.replace('_results', '')
            
            try:
                with open(result_file, 'rb') as f:
                    self.results[job_id] = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load results for {job_id}: {e}")
        
        logger.info(f"Loaded results for {len(self.results)} jobs")
        return self.results
    
    def aggregate_metrics(self, metric_path: str) -> Dict[str, Any]:
        """Aggregate metrics across all jobs."""
        if not self.results:
            self.load_results()
        
        values = []
        for job_results in self.results.values():
            try:
                # Navigate nested dictionary using dot notation
                value = job_results
                for key in metric_path.split('.'):
                    value = value[key]
                values.append(value)
            except (KeyError, TypeError):
                continue
        
        if not values:
            return {}
        
        # Calculate statistics
        values = np.array(values)
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'count': len(values)
        }
    
    def generate_report(self, output_path: Union[str, Path]):
        """Generate comprehensive analysis report."""
        if not self.results:
            self.load_results()
        
        report = {
            'batch_summary': self.summary,
            'job_count': len(self.results),
            'metrics': {}
        }
        
        # Common metrics to aggregate
        common_metrics = [
            'simulation_results.total_energy',
            'simulation_results.peak_power', 
            'training_history.loss',
            'training_history.accuracy',
            'analysis_results.performance.inference_time',
            'analysis_results.performance.energy_efficiency'
        ]
        
        for metric in common_metrics:
            aggregated = self.aggregate_metrics(metric)
            if aggregated:
                report['metrics'][metric] = aggregated
        
        # Save report
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analysis report saved to {output_path}")
        return report