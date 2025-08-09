"""
Advanced distributed computing and scalable batch processing for PhoMem-CoSim.
"""

import asyncio
import time
import uuid
import json
import pickle
import threading
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from queue import Queue, Empty
import logging
import traceback

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import dask
    import dask.distributed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    from kubernetes import client, config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

@dataclass
class Task:
    """Distributed task definition."""
    task_id: str
    function_name: str
    args: Tuple
    kwargs: Dict[str, Any]
    priority: int = 0
    timeout: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    status: str  # 'completed', 'failed', 'timeout'
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    worker_id: Optional[str] = None
    completed_at: float = field(default_factory=time.time)

class DistributedTaskQueue:
    """Distributed task queue using Redis or in-memory fallback."""
    
    def __init__(self, redis_url: Optional[str] = None, max_retries: int = 3):
        self.max_retries = max_retries
        self.pending_tasks = {}
        self.results = {}
        
        # Initialize Redis if available
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()  # Test connection
                self.use_redis = True
                self.task_queue = 'phomem:tasks'
                self.result_queue = 'phomem:results'
            except Exception as e:
                logging.warning(f"Redis connection failed: {e}, using in-memory queue")
                self.use_redis = False
                self._init_memory_queue()
        else:
            self.use_redis = False
            self._init_memory_queue()
    
    def _init_memory_queue(self):
        """Initialize in-memory queue."""
        self.memory_queue = Queue()
        self.memory_results = {}
        self.redis_client = None
    
    def submit_task(self, task: Task) -> str:
        """Submit task to distributed queue."""
        if self.use_redis:
            task_data = {
                'task_id': task.task_id,
                'function_name': task.function_name,
                'args': pickle.dumps(task.args),
                'kwargs': pickle.dumps(task.kwargs),
                'priority': task.priority,
                'timeout': task.timeout,
                'created_at': task.created_at,
                'metadata': json.dumps(task.metadata)
            }
            
            # Add to priority queue (higher priority = lower score)
            self.redis_client.zadd(self.task_queue, {json.dumps(task_data): -task.priority})
            
        else:
            self.memory_queue.put((task.priority, task))
        
        self.pending_tasks[task.task_id] = task
        return task.task_id
    
    def get_task(self, worker_id: str, timeout: float = 1.0) -> Optional[Task]:
        """Get next task from queue."""
        if self.use_redis:
            # Get highest priority task (lowest score)
            result = self.redis_client.bzpopmin(self.task_queue, timeout=timeout)
            if result:
                _, task_data_json, _ = result
                task_data = json.loads(task_data_json)
                
                task = Task(
                    task_id=task_data['task_id'],
                    function_name=task_data['function_name'],
                    args=pickle.loads(task_data['args']),
                    kwargs=pickle.loads(task_data['kwargs']),
                    priority=task_data['priority'],
                    timeout=task_data['timeout'],
                    created_at=task_data['created_at'],
                    metadata=json.loads(task_data['metadata'])
                )
                return task
        else:
            try:
                _, task = self.memory_queue.get(timeout=timeout)
                return task
            except Empty:
                pass
        
        return None
    
    def submit_result(self, result: TaskResult):
        """Submit task result."""
        if self.use_redis:
            result_data = {
                'task_id': result.task_id,
                'status': result.status,
                'result': pickle.dumps(result.result) if result.result is not None else None,
                'error': result.error,
                'execution_time': result.execution_time,
                'worker_id': result.worker_id,
                'completed_at': result.completed_at
            }
            
            self.redis_client.hset(self.result_queue, result.task_id, json.dumps(result_data))
            # Set expiration for results (24 hours)
            self.redis_client.expire(f"{self.result_queue}:{result.task_id}", 86400)
        else:
            self.memory_results[result.task_id] = result
        
        # Clean up pending task
        if result.task_id in self.pending_tasks:
            del self.pending_tasks[result.task_id]
    
    def get_result(self, task_id: str, timeout: float = None) -> Optional[TaskResult]:
        """Get task result."""
        start_time = time.time()
        
        while timeout is None or (time.time() - start_time) < timeout:
            if self.use_redis:
                result_data = self.redis_client.hget(self.result_queue, task_id)
                if result_data:
                    data = json.loads(result_data)
                    return TaskResult(
                        task_id=data['task_id'],
                        status=data['status'],
                        result=pickle.loads(data['result']) if data['result'] else None,
                        error=data['error'],
                        execution_time=data['execution_time'],
                        worker_id=data['worker_id'],
                        completed_at=data['completed_at']
                    )
            else:
                if task_id in self.memory_results:
                    return self.memory_results[task_id]
            
            if timeout is None:
                break
            time.sleep(0.1)
        
        return None

class DistributedWorker:
    """Distributed worker for task execution."""
    
    def __init__(self, 
                 worker_id: str = None,
                 function_registry: Dict[str, Callable] = None,
                 task_queue: DistributedTaskQueue = None):
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.function_registry = function_registry or {}
        self.task_queue = task_queue or DistributedTaskQueue()
        self.running = False
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'start_time': None
        }
    
    def register_function(self, name: str, func: Callable):
        """Register function for task execution."""
        self.function_registry[name] = func
    
    def start(self):
        """Start worker loop."""
        self.running = True
        self.stats['start_time'] = time.time()
        
        logging.info(f"Worker {self.worker_id} started")
        
        while self.running:
            try:
                # Get next task
                task = self.task_queue.get_task(self.worker_id, timeout=1.0)
                if not task:
                    continue
                
                # Execute task
                result = self._execute_task(task)
                
                # Submit result
                self.task_queue.submit_result(result)
                
                # Update stats
                if result.status == 'completed':
                    self.stats['tasks_completed'] += 1
                else:
                    self.stats['tasks_failed'] += 1
                self.stats['total_execution_time'] += result.execution_time
                
            except KeyboardInterrupt:
                logging.info(f"Worker {self.worker_id} interrupted")
                break
            except Exception as e:
                logging.error(f"Worker {self.worker_id} error: {e}")
                time.sleep(1.0)  # Backoff on error
        
        logging.info(f"Worker {self.worker_id} stopped")
    
    def stop(self):
        """Stop worker."""
        self.running = False
    
    def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task."""
        start_time = time.time()
        
        try:
            # Check if function is registered
            if task.function_name not in self.function_registry:
                raise ValueError(f"Function '{task.function_name}' not registered")
            
            func = self.function_registry[task.function_name]
            
            # Execute with timeout if specified
            if task.timeout:
                result = self._execute_with_timeout(func, task.args, task.kwargs, task.timeout)
            else:
                result = func(*task.args, **task.kwargs)
            
            execution_time = time.time() - start_time
            
            return TaskResult(
                task_id=task.task_id,
                status='completed',
                result=result,
                execution_time=execution_time,
                worker_id=self.worker_id
            )
            
        except TimeoutError:
            return TaskResult(
                task_id=task.task_id,
                status='timeout',
                error='Task exceeded timeout',
                execution_time=time.time() - start_time,
                worker_id=self.worker_id
            )
        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                status='failed',
                error=str(e),
                execution_time=time.time() - start_time,
                worker_id=self.worker_id
            )
    
    def _execute_with_timeout(self, func: Callable, args: Tuple, kwargs: Dict, timeout: float):
        """Execute function with timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Task execution timeout")
        
        # Set timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel timeout
            return result
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        stats = self.stats.copy()
        if stats['start_time']:
            stats['uptime'] = time.time() - stats['start_time']
            if stats['tasks_completed'] > 0:
                stats['avg_execution_time'] = stats['total_execution_time'] / stats['tasks_completed']
        return stats

class ScalableSimulationManager:
    """Scalable simulation manager for large-scale parameter studies."""
    
    def __init__(self, 
                 task_queue: DistributedTaskQueue = None,
                 use_ray: bool = False,
                 use_dask: bool = False):
        self.task_queue = task_queue or DistributedTaskQueue()
        self.workers = []
        self.results = {}
        
        # Initialize distributed computing framework
        self.use_ray = use_ray and RAY_AVAILABLE
        self.use_dask = use_dask and DASK_AVAILABLE
        
        if self.use_ray:
            try:
                ray.init(ignore_reinit_error=True)
                logging.info("Ray initialized for distributed computing")
            except Exception as e:
                logging.warning(f"Ray initialization failed: {e}")
                self.use_ray = False
        
        if self.use_dask:
            try:
                self.dask_client = dask.distributed.Client(processes=False)
                logging.info("Dask client initialized")
            except Exception as e:
                logging.warning(f"Dask initialization failed: {e}")
                self.use_dask = False
    
    def start_local_workers(self, num_workers: int = None, function_registry: Dict[str, Callable] = None):
        """Start local worker processes."""
        num_workers = num_workers or mp.cpu_count()
        
        for i in range(num_workers):
            worker = DistributedWorker(
                worker_id=f"local_worker_{i}",
                function_registry=function_registry,
                task_queue=self.task_queue
            )
            
            # Start worker in separate process
            process = mp.Process(target=worker.start)
            process.daemon = True
            process.start()
            
            self.workers.append({
                'worker': worker,
                'process': process,
                'worker_id': worker.worker_id
            })
        
        logging.info(f"Started {num_workers} local workers")
    
    def submit_simulation_batch(self, 
                               simulation_configs: List[Dict[str, Any]],
                               function_name: str = 'run_simulation',
                               priority: int = 0) -> List[str]:
        """Submit batch of simulations for distributed execution."""
        task_ids = []
        
        for i, config in enumerate(simulation_configs):
            task_id = f"sim_{uuid.uuid4().hex[:8]}"
            
            task = Task(
                task_id=task_id,
                function_name=function_name,
                args=(config,),
                kwargs={},
                priority=priority,
                metadata={'batch_index': i}
            )
            
            self.task_queue.submit_task(task)
            task_ids.append(task_id)
        
        logging.info(f"Submitted {len(task_ids)} simulations to queue")
        return task_ids
    
    def wait_for_results(self, task_ids: List[str], timeout: float = None) -> Dict[str, TaskResult]:
        """Wait for simulation results."""
        results = {}
        start_time = time.time()
        
        while len(results) < len(task_ids):
            for task_id in task_ids:
                if task_id in results:
                    continue
                
                result = self.task_queue.get_result(task_id, timeout=1.0)
                if result:
                    results[task_id] = result
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                logging.warning(f"Timeout waiting for results: {len(results)}/{len(task_ids)} completed")
                break
            
            time.sleep(0.1)
        
        return results
    
    def run_parameter_sweep(self,
                           base_config: Dict[str, Any],
                           parameter_grid: Dict[str, List[Any]],
                           function_name: str = 'run_simulation',
                           max_concurrent: int = None) -> List[TaskResult]:
        """Run distributed parameter sweep."""
        import itertools
        
        # Generate parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        configs = []
        for combination in itertools.product(*param_values):
            config = base_config.copy()
            param_dict = dict(zip(param_names, combination))
            config.update(param_dict)
            configs.append(config)
        
        logging.info(f"Running parameter sweep: {len(configs)} configurations")
        
        # Submit tasks in batches if max_concurrent is specified
        if max_concurrent and len(configs) > max_concurrent:
            all_results = []
            
            for i in range(0, len(configs), max_concurrent):
                batch = configs[i:i + max_concurrent]
                task_ids = self.submit_simulation_batch(batch, function_name)
                batch_results = self.wait_for_results(task_ids)
                all_results.extend(batch_results.values())
                
                logging.info(f"Completed batch {i//max_concurrent + 1}/{(len(configs) + max_concurrent - 1)//max_concurrent}")
            
            return all_results
        else:
            task_ids = self.submit_simulation_batch(configs, function_name)
            results = self.wait_for_results(task_ids)
            return list(results.values())
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get distributed cluster status."""
        status = {
            'local_workers': len(self.workers),
            'active_workers': sum(1 for w in self.workers if w['process'].is_alive()),
            'task_queue_size': 0,  # Would need to implement queue size check
            'frameworks': {
                'ray': self.use_ray,
                'dask': self.use_dask,
                'redis': self.task_queue.use_redis if hasattr(self.task_queue, 'use_redis') else False
            }
        }
        
        if self.use_ray:
            try:
                ray_status = ray.cluster_resources()
                status['ray_resources'] = ray_status
            except:
                pass
        
        if self.use_dask:
            try:
                status['dask_workers'] = len(self.dask_client.nthreads())
            except:
                pass
        
        return status
    
    def shutdown(self):
        """Shutdown distributed computing resources."""
        # Stop local workers
        for worker_info in self.workers:
            worker_info['process'].terminate()
        
        # Shutdown Ray
        if self.use_ray:
            try:
                ray.shutdown()
            except:
                pass
        
        # Close Dask client
        if self.use_dask:
            try:
                self.dask_client.close()
            except:
                pass
        
        logging.info("Distributed computing resources shut down")

class KubernetesJobManager:
    """Kubernetes job management for cloud-scale simulations."""
    
    def __init__(self, namespace: str = 'phomem'):
        if not KUBERNETES_AVAILABLE:
            raise ImportError("Kubernetes client not available")
        
        self.namespace = namespace
        try:
            config.load_incluster_config()  # Try in-cluster config first
        except:
            config.load_kube_config()  # Fall back to local config
        
        self.batch_v1 = client.BatchV1Api()
        self.core_v1 = client.CoreV1Api()
    
    def create_simulation_job(self,
                             job_name: str,
                             image: str,
                             command: List[str],
                             parallelism: int = 1,
                             completions: int = 1,
                             resources: Dict[str, str] = None) -> str:
        """Create Kubernetes job for simulation."""
        
        resources = resources or {'cpu': '1', 'memory': '2Gi'}
        
        # Job specification
        job_spec = client.V1Job(
            metadata=client.V1ObjectMeta(name=job_name, namespace=self.namespace),
            spec=client.V1JobSpec(
                parallelism=parallelism,
                completions=completions,
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name='simulation',
                                image=image,
                                command=command,
                                resources=client.V1ResourceRequirements(
                                    requests=resources,
                                    limits=resources
                                )
                            )
                        ],
                        restart_policy='Never'
                    )
                )
            )
        )
        
        # Create job
        response = self.batch_v1.create_namespaced_job(
            namespace=self.namespace,
            body=job_spec
        )
        
        logging.info(f"Created Kubernetes job: {job_name}")
        return response.metadata.name
    
    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        """Get job status."""
        try:
            job = self.batch_v1.read_namespaced_job(name=job_name, namespace=self.namespace)
            
            return {
                'name': job.metadata.name,
                'active': job.status.active or 0,
                'succeeded': job.status.succeeded or 0,
                'failed': job.status.failed or 0,
                'start_time': job.status.start_time,
                'completion_time': job.status.completion_time,
                'conditions': [
                    {
                        'type': condition.type,
                        'status': condition.status,
                        'reason': condition.reason
                    } for condition in (job.status.conditions or [])
                ]
            }
        except Exception as e:
            logging.error(f"Failed to get job status: {e}")
            return {'error': str(e)}
    
    def delete_job(self, job_name: str):
        """Delete job and associated pods."""
        try:
            # Delete job
            self.batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                body=client.V1DeleteOptions(propagation_policy='Foreground')
            )
            
            logging.info(f"Deleted Kubernetes job: {job_name}")
        except Exception as e:
            logging.error(f"Failed to delete job: {e}")

# Global instances for easy access
_task_queue = None
_simulation_manager = None

def get_task_queue(redis_url: str = None) -> DistributedTaskQueue:
    """Get global task queue instance."""
    global _task_queue
    if _task_queue is None:
        _task_queue = DistributedTaskQueue(redis_url=redis_url)
    return _task_queue

def get_simulation_manager(use_ray: bool = False, use_dask: bool = False) -> ScalableSimulationManager:
    """Get global simulation manager instance."""
    global _simulation_manager
    if _simulation_manager is None:
        _simulation_manager = ScalableSimulationManager(use_ray=use_ray, use_dask=use_dask)
    return _simulation_manager

def run_distributed_simulations(configs: List[Dict[str, Any]], 
                               num_workers: int = None,
                               function_registry: Dict[str, Callable] = None) -> List[TaskResult]:
    """Convenience function to run distributed simulations."""
    
    # Get or create simulation manager
    manager = get_simulation_manager()
    
    # Start local workers if none exist
    if not manager.workers:
        manager.start_local_workers(num_workers, function_registry)
    
    # Submit and wait for results
    task_ids = manager.submit_simulation_batch(configs)
    results = manager.wait_for_results(task_ids)
    
    return list(results.values())