"""
Advanced performance optimization and scaling framework for Generation 3.
"""

import jax
import jax.numpy as jnp
import numpy as np
import threading
import multiprocessing
import time
import logging
import functools
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import warnings
import psutil
import gc

from .utils.logging import get_logger
from .utils.exceptions import PhoMemError

logger = get_logger('performance')


class PerformanceError(PhoMemError):
    """Performance-related errors."""
    pass


class ResourceError(PhoMemError):
    """Resource-related errors."""
    pass


class AdaptiveCache:
    """Adaptive caching system with performance-based eviction."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600.0):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.hit_counts = {}
        self.memory_usage = {}
        self.lock = threading.RLock()
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _hash_key(self, key) -> str:
        """Create hashable key from various input types."""
        if isinstance(key, (tuple, list)):
            return str(hash(tuple(str(k) for k in key)))
        elif hasattr(key, 'tobytes'):  # numpy/jax arrays
            return str(hash(key.tobytes()))
        else:
            return str(hash(str(key)))
    
    def _estimate_memory(self, value) -> int:
        """Estimate memory usage of cached value."""
        if hasattr(value, 'nbytes'):
            return value.nbytes
        elif isinstance(value, (list, tuple)):
            return sum(self._estimate_memory(v) for v in value)
        elif isinstance(value, dict):
            return sum(self._estimate_memory(k) + self._estimate_memory(v) 
                      for k, v in value.items())
        else:
            return len(str(value).encode('utf-8'))
    
    def get(self, key, default=None):
        """Get value from cache with adaptive performance tracking."""
        with self.lock:
            hash_key = self._hash_key(key)
            current_time = time.time()
            
            if hash_key in self.cache:
                # Check TTL
                if current_time - self.access_times.get(hash_key, 0) < self.ttl:
                    self.access_times[hash_key] = current_time
                    self.hit_counts[hash_key] = self.hit_counts.get(hash_key, 0) + 1
                    self.hits += 1
                    return self.cache[hash_key]
                else:
                    # Expired
                    self._remove_key(hash_key)
            
            self.misses += 1
            return default
    
    def put(self, key, value):
        """Store value in cache with intelligent eviction."""
        with self.lock:
            hash_key = self._hash_key(key)
            current_time = time.time()
            memory_usage = self._estimate_memory(value)
            
            # Evict if necessary
            while len(self.cache) >= self.max_size:
                self._evict_least_valuable()
            
            # Store new value
            self.cache[hash_key] = value
            self.access_times[hash_key] = current_time
            self.hit_counts[hash_key] = 0
            self.memory_usage[hash_key] = memory_usage
    
    def _remove_key(self, hash_key: str):
        """Remove key from all tracking structures."""
        self.cache.pop(hash_key, None)
        self.access_times.pop(hash_key, None)
        self.hit_counts.pop(hash_key, None)
        self.memory_usage.pop(hash_key, None)
    
    def _evict_least_valuable(self):
        """Evict least valuable item based on access pattern and memory."""
        if not self.cache:
            return
        
        current_time = time.time()
        min_score = float('inf')
        evict_key = None
        
        for hash_key in self.cache:
            # Value score based on hits, recency, and memory efficiency
            hits = self.hit_counts.get(hash_key, 0)
            recency = current_time - self.access_times.get(hash_key, 0)
            memory = self.memory_usage.get(hash_key, 1)
            
            # Lower is better for eviction
            score = memory / max(1, hits) + recency / 3600.0
            
            if score < min_score:
                min_score = score
                evict_key = hash_key
        
        if evict_key:
            self._remove_key(evict_key)
            self.evictions += 1
            logger.debug(f"Evicted cache key with score {min_score:.2f}")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(1, total_requests)
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate,
            'total_memory_mb': sum(self.memory_usage.values()) / (1024 * 1024)
        }
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_counts.clear()
            self.memory_usage.clear()


class ResourceMonitor:
    """System resource monitoring and management."""
    
    def __init__(self):
        self.cpu_threshold = 80.0  # %
        self.memory_threshold = 85.0  # %
        self.monitoring = False
        self.alerts = []
    
    def get_system_stats(self) -> Dict[str, float]:
        """Get current system resource utilization."""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'active_threads': threading.active_count(),
                'process_count': len(psutil.pids())
            }
        except Exception as e:
            logger.warning(f"Failed to get system stats: {e}")
            return {}
    
    def check_resource_limits(self) -> List[str]:
        """Check if system resources are within limits."""
        warnings_list = []
        stats = self.get_system_stats()
        
        if stats.get('cpu_percent', 0) > self.cpu_threshold:
            warnings_list.append(f"High CPU usage: {stats['cpu_percent']:.1f}%")
        
        if stats.get('memory_percent', 0) > self.memory_threshold:
            warnings_list.append(f"High memory usage: {stats['memory_percent']:.1f}%")
        
        if stats.get('available_memory_gb', 10) < 1.0:
            warnings_list.append(f"Low available memory: {stats['available_memory_gb']:.2f} GB")
        
        return warnings_list
    
    def auto_gc_if_needed(self):
        """Trigger garbage collection if memory usage is high."""
        stats = self.get_system_stats()
        if stats.get('memory_percent', 0) > 80.0:
            logger.info("High memory usage detected, triggering garbage collection")
            gc.collect()
            # Clear JAX compilation cache if available
            try:
                jax.clear_caches()
                logger.debug("Cleared JAX compilation cache")
            except Exception:
                pass


class BatchProcessor:
    """Intelligent batch processing with adaptive sizing."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.resource_monitor = ResourceMonitor()
        self.adaptive_batch_size = 32
        self.performance_history = []
    
    def _adaptive_batch_sizing(self, data_size: int) -> int:
        """Determine optimal batch size based on system resources and history."""
        system_stats = self.resource_monitor.get_system_stats()
        
        # Adjust based on available memory
        available_gb = system_stats.get('available_memory_gb', 4.0)
        if available_gb < 2.0:
            target_batch_size = max(8, self.adaptive_batch_size // 2)
        elif available_gb > 8.0:
            target_batch_size = min(128, self.adaptive_batch_size * 2)
        else:
            target_batch_size = self.adaptive_batch_size
        
        # Adjust based on CPU usage
        cpu_usage = system_stats.get('cpu_percent', 50.0)
        if cpu_usage > 90.0:
            target_batch_size = max(8, target_batch_size // 2)
        
        # Don't exceed data size
        return min(target_batch_size, data_size)
    
    def process_batches(self, 
                       data: List[Any],
                       process_func: Callable,
                       use_processes: bool = False,
                       progress_callback: Optional[Callable] = None) -> List[Any]:
        """Process data in adaptive batches with optimal parallelization."""
        
        if not data:
            return []
        
        batch_size = self._adaptive_batch_sizing(len(data))
        logger.info(f"Processing {len(data)} items in batches of {batch_size}")
        
        # Create batches
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        results = []
        
        start_time = time.time()
        
        # Choose execution strategy
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        try:
            with executor_class(max_workers=self.max_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(self._process_batch, batch, process_func): i 
                    for i, batch in enumerate(batches)
                }
                
                # Collect results as they complete
                batch_results = [None] * len(batches)
                completed = 0
                
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_result = future.result()
                        batch_results[batch_idx] = batch_result
                        completed += 1
                        
                        if progress_callback:
                            progress_callback(completed, len(batches))
                        
                        # Resource monitoring
                        warnings_list = self.resource_monitor.check_resource_limits()
                        if warnings_list:
                            logger.warning(f"Resource warnings: {warnings_list}")
                        
                    except Exception as e:
                        logger.error(f"Batch {batch_idx} failed: {e}")
                        batch_results[batch_idx] = []
                
                # Flatten results maintaining order
                for batch_result in batch_results:
                    if batch_result is not None:
                        results.extend(batch_result)
        
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise PerformanceError(f"Batch processing failed: {e}")
        
        # Update performance tracking
        total_time = time.time() - start_time
        throughput = len(data) / total_time if total_time > 0 else 0
        
        self.performance_history.append({
            'batch_size': batch_size,
            'total_items': len(data),
            'total_time': total_time,
            'throughput': throughput
        })
        
        # Adaptive learning
        self._update_adaptive_batch_size(throughput)
        
        logger.info(f"Processed {len(data)} items in {total_time:.2f}s (throughput: {throughput:.1f} items/s)")
        
        return results
    
    def _process_batch(self, batch: List[Any], process_func: Callable) -> List[Any]:
        """Process a single batch of data."""
        try:
            return [process_func(item) for item in batch]
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return []
    
    def _update_adaptive_batch_size(self, current_throughput: float):
        """Update adaptive batch size based on performance feedback."""
        if len(self.performance_history) < 2:
            return
        
        recent_performance = self.performance_history[-5:]  # Last 5 runs
        avg_throughput = np.mean([p['throughput'] for p in recent_performance])
        
        if current_throughput > avg_throughput * 1.1:
            # Performance improved, try larger batches
            self.adaptive_batch_size = min(256, int(self.adaptive_batch_size * 1.2))
        elif current_throughput < avg_throughput * 0.9:
            # Performance degraded, try smaller batches
            self.adaptive_batch_size = max(8, int(self.adaptive_batch_size * 0.8))


class JITOptimizer:
    """JAX JIT compilation optimization and caching."""
    
    def __init__(self):
        self.compilation_cache = {}
        self.compilation_times = {}
        self.function_stats = {}
    
    def smart_jit(self, func: Callable, static_argnums: Tuple = (), **jit_kwargs):
        """Intelligent JIT compilation with caching and optimization."""
        
        # Generate cache key
        cache_key = (
            func.__name__,
            static_argnums,
            tuple(sorted(jit_kwargs.items()))
        )
        
        if cache_key in self.compilation_cache:
            logger.debug(f"Using cached JIT function: {func.__name__}")
            return self.compilation_cache[cache_key]
        
        logger.info(f"Compiling JIT function: {func.__name__}")
        start_time = time.time()
        
        try:
            # Apply JIT with optimizations
            jit_func = jax.jit(func, static_argnums=static_argnums, **jit_kwargs)
            
            # Warm up compilation
            self._warm_up_function(jit_func, func.__name__)
            
            compilation_time = time.time() - start_time
            
            # Cache compiled function
            self.compilation_cache[cache_key] = jit_func
            self.compilation_times[cache_key] = compilation_time
            
            # Track function stats
            self.function_stats[func.__name__] = {
                'compilation_time': compilation_time,
                'cache_key': cache_key,
                'call_count': 0,
                'total_runtime': 0.0
            }
            
            logger.info(f"JIT compilation completed in {compilation_time:.2f}s: {func.__name__}")
            
            return jit_func
            
        except Exception as e:
            logger.error(f"JIT compilation failed for {func.__name__}: {e}")
            # Fallback to non-JIT version
            return func
    
    def _warm_up_function(self, jit_func: Callable, func_name: str):
        """Warm up JIT function with dummy inputs if possible."""
        try:
            # This is a placeholder - in real implementation, we'd need
            # function signature analysis to generate appropriate dummy inputs
            logger.debug(f"Warming up JIT function: {func_name}")
        except Exception:
            pass  # Warm-up is optional
    
    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get JIT compilation performance statistics."""
        total_compilation_time = sum(self.compilation_times.values())
        
        return {
            'cached_functions': len(self.compilation_cache),
            'total_compilation_time': total_compilation_time,
            'function_stats': self.function_stats,
            'average_compilation_time': (
                total_compilation_time / len(self.compilation_times) 
                if self.compilation_times else 0
            )
        }


class PerformanceProfiler:
    """Advanced performance profiling and optimization suggestions."""
    
    def __init__(self):
        self.profiles = {}
        self.bottlenecks = []
        self.optimization_suggestions = []
    
    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling operations."""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.virtual_memory().used
            
            profile_data = {
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'timestamp': start_time
            }
            
            if operation_name not in self.profiles:
                self.profiles[operation_name] = []
            
            self.profiles[operation_name].append(profile_data)
            
            # Analyze for bottlenecks
            self._analyze_performance(operation_name, profile_data)
    
    def _analyze_performance(self, operation_name: str, profile_data: Dict):
        """Analyze performance data and generate optimization suggestions."""
        duration = profile_data['duration']
        memory_delta = profile_data['memory_delta']
        
        # Identify slow operations
        if duration > 1.0:  # > 1 second
            self.bottlenecks.append({
                'operation': operation_name,
                'type': 'slow_execution',
                'duration': duration,
                'timestamp': profile_data['timestamp']
            })
            
            self.optimization_suggestions.append(
                f"Operation '{operation_name}' took {duration:.2f}s - consider optimization"
            )
        
        # Identify memory-intensive operations
        if memory_delta > 100 * 1024 * 1024:  # > 100MB
            self.bottlenecks.append({
                'operation': operation_name,
                'type': 'high_memory',
                'memory_mb': memory_delta / (1024 * 1024),
                'timestamp': profile_data['timestamp']
            })
            
            self.optimization_suggestions.append(
                f"Operation '{operation_name}' used {memory_delta / (1024**2):.1f}MB - check memory efficiency"
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        operation_summaries = {}
        
        for operation, profiles in self.profiles.items():
            durations = [p['duration'] for p in profiles]
            memory_deltas = [p['memory_delta'] for p in profiles]
            
            operation_summaries[operation] = {
                'call_count': len(profiles),
                'avg_duration': np.mean(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations),
                'std_duration': np.std(durations),
                'avg_memory_delta_mb': np.mean(memory_deltas) / (1024**2),
                'total_time': np.sum(durations)
            }
        
        return {
            'operation_summaries': operation_summaries,
            'bottlenecks': self.bottlenecks,
            'optimization_suggestions': self.optimization_suggestions,
            'total_operations': sum(len(profiles) for profiles in self.profiles.values())
        }


class AutoScaler:
    """Automatic scaling based on workload and system resources."""
    
    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.scaling_history = []
        self.current_scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 4.0
    
    def auto_scale_computation(self, 
                              base_computation: Callable,
                              workload_size: int,
                              **kwargs) -> Any:
        """Automatically scale computation based on system resources."""
        
        # Assess current system state
        system_stats = self.resource_monitor.get_system_stats()
        optimal_scale = self._calculate_optimal_scale(system_stats, workload_size)
        
        if optimal_scale != self.current_scale:
            logger.info(f"Scaling computation from {self.current_scale:.2f}x to {optimal_scale:.2f}x")
            self.current_scale = optimal_scale
        
        # Apply scaling to computation
        scaled_result = self._apply_scaling(base_computation, optimal_scale, **kwargs)
        
        # Track scaling performance
        self.scaling_history.append({
            'timestamp': time.time(),
            'scale_factor': optimal_scale,
            'workload_size': workload_size,
            'system_stats': system_stats
        })
        
        return scaled_result
    
    def _calculate_optimal_scale(self, system_stats: Dict, workload_size: int) -> float:
        """Calculate optimal scaling factor based on system resources."""
        cpu_usage = system_stats.get('cpu_percent', 50.0)
        memory_usage = system_stats.get('memory_percent', 50.0)
        available_memory_gb = system_stats.get('available_memory_gb', 4.0)
        
        # Start with base scale
        optimal_scale = 1.0
        
        # Scale down if resources are constrained
        if cpu_usage > 90.0 or memory_usage > 90.0:
            optimal_scale *= 0.5
        elif cpu_usage > 80.0 or memory_usage > 80.0:
            optimal_scale *= 0.75
        
        # Scale up if resources are abundant
        elif cpu_usage < 30.0 and memory_usage < 50.0 and available_memory_gb > 8.0:
            optimal_scale *= 2.0
        elif cpu_usage < 50.0 and memory_usage < 70.0 and available_memory_gb > 4.0:
            optimal_scale *= 1.5
        
        # Adjust for workload size
        if workload_size > 10000:
            optimal_scale *= 1.2  # Larger workloads benefit from scaling
        elif workload_size < 100:
            optimal_scale *= 0.8  # Smaller workloads don't need full resources
        
        # Clamp to limits
        return max(self.min_scale, min(self.max_scale, optimal_scale))
    
    def _apply_scaling(self, computation: Callable, scale_factor: float, **kwargs) -> Any:
        """Apply scaling factor to computation."""
        if scale_factor < 1.0:
            # Scale down: reduce precision, simplify computation
            logger.debug(f"Scaling down computation by {scale_factor:.2f}x")
            # Implementation would depend on specific computation type
            return computation(**kwargs)
        elif scale_factor > 1.0:
            # Scale up: increase parallelism, use more resources
            logger.debug(f"Scaling up computation by {scale_factor:.2f}x")
            # Implementation would depend on specific computation type
            return computation(**kwargs)
        else:
            return computation(**kwargs)


# Global performance optimization instances
_global_cache = AdaptiveCache(max_size=2000, ttl=7200)  # 2 hour TTL
_global_batch_processor = BatchProcessor()
_global_jit_optimizer = JITOptimizer()
_global_profiler = PerformanceProfiler()
_global_autoscaler = AutoScaler()


def get_cache() -> AdaptiveCache:
    """Get global cache instance."""
    return _global_cache


def get_batch_processor() -> BatchProcessor:
    """Get global batch processor instance."""
    return _global_batch_processor


def get_jit_optimizer() -> JITOptimizer:
    """Get global JIT optimizer instance."""
    return _global_jit_optimizer


def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance."""
    return _global_profiler


def get_autoscaler() -> AutoScaler:
    """Get global autoscaler instance."""
    return _global_autoscaler


# Performance optimization decorators
def cached(ttl: float = 3600.0, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        cache = get_cache()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = (func.__name__, args, tuple(sorted(kwargs.items())))
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


def profiled(operation_name: Optional[str] = None):
    """Decorator for performance profiling."""
    def decorator(func: Callable) -> Callable:
        profiler = get_profiler()
        op_name = operation_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with profiler.profile(op_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def auto_jit(static_argnums: Tuple = (), **jit_kwargs):
    """Decorator for automatic JIT optimization."""
    def decorator(func: Callable) -> Callable:
        jit_optimizer = get_jit_optimizer()
        return jit_optimizer.smart_jit(func, static_argnums, **jit_kwargs)
    
    return decorator


def batch_process(batch_size: Optional[int] = None, use_processes: bool = False):
    """Decorator for batch processing of iterable inputs."""
    def decorator(func: Callable) -> Callable:
        batch_processor = get_batch_processor()
        
        @functools.wraps(func)
        def wrapper(data: List[Any], *args, **kwargs):
            if not isinstance(data, (list, tuple)):
                return func(data, *args, **kwargs)
            
            def process_item(item):
                return func(item, *args, **kwargs)
            
            return batch_processor.process_batches(
                data, process_item, use_processes=use_processes
            )
        
        return wrapper
    return decorator


def resource_monitor(memory_limit_gb: Optional[float] = None, 
                    cpu_limit_percent: Optional[float] = None):
    """Decorator for resource monitoring and limits."""
    def decorator(func: Callable) -> Callable:
        monitor = ResourceMonitor()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check resource limits before execution
            stats = monitor.get_system_stats()
            
            if memory_limit_gb and stats.get('memory_percent', 0) > 95.0:
                raise ResourceError(f"Memory usage too high: {stats['memory_percent']:.1f}%")
            
            if cpu_limit_percent and stats.get('cpu_percent', 0) > cpu_limit_percent:
                warnings.warn(f"High CPU usage: {stats['cpu_percent']:.1f}%")
            
            # Execute with monitoring
            try:
                result = func(*args, **kwargs)
                monitor.auto_gc_if_needed()  # Cleanup if needed
                return result
            except MemoryError:
                monitor.auto_gc_if_needed()
                raise ResourceError("Out of memory during execution")
        
        return wrapper
    return decorator


# Performance analysis utilities
def analyze_performance() -> Dict[str, Any]:
    """Get comprehensive performance analysis."""
    cache_stats = get_cache().stats()
    jit_stats = get_jit_optimizer().get_compilation_stats()
    profiler_stats = get_profiler().get_performance_summary()
    
    return {
        'cache_performance': cache_stats,
        'jit_performance': jit_stats,
        'operation_performance': profiler_stats,
        'timestamp': time.time()
    }


def optimize_for_workload(workload_type: str = 'balanced'):
    """Configure optimizations for specific workload types."""
    cache = get_cache()
    
    if workload_type == 'memory_intensive':
        cache.max_size = 500  # Smaller cache
        cache.ttl = 1800  # Shorter TTL
    elif workload_type == 'compute_intensive':
        cache.max_size = 5000  # Larger cache
        cache.ttl = 7200  # Longer TTL
    elif workload_type == 'io_intensive':
        cache.max_size = 1000
        cache.ttl = 3600
    
    logger.info(f"Optimized configuration for {workload_type} workload")