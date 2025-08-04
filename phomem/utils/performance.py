"""
Performance optimization utilities for PhoMem-CoSim.
"""

import time
import functools
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, Any, Callable, Optional, List, Union
import pickle
import hashlib
import os
from pathlib import Path
import gc
import weakref

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from .logging import get_logger
from .exceptions import PhoMemError


class PerformanceOptimizer:
    """Performance optimization and monitoring utilities."""
    
    def __init__(self):
        self.logger = get_logger('performance')
        self.cache_dir = Path.home() / '.phomem' / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.call_stats = {}
        self.memory_stats = {}
        
        # JIT compilation cache
        self.jit_cache = weakref.WeakValueDictionary()
        
        # Thread pool for concurrent operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = None  # Lazy initialization
    
    def __del__(self):
        """Cleanup thread pools."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)
        if hasattr(self, 'process_pool') and self.process_pool:
            self.process_pool.shutdown(wait=False)
    
    def memoize(self, cache_file: str = None, ttl: int = 3600):
        """Decorator for memoizing expensive function calls."""
        def decorator(func):
            cache = {}
            cache_times = {}
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key
                key = self._create_cache_key(args, kwargs)
                
                # Check if cached result is still valid
                if key in cache:
                    if ttl <= 0 or (time.time() - cache_times.get(key, 0)) < ttl:
                        self.logger.debug(f"Cache hit for {func.__name__}")
                        return cache[key]
                    else:
                        # Expired, remove from cache
                        del cache[key]
                        del cache_times[key]
                
                # Compute result and cache it
                start_time = time.time()
                result = func(*args, **kwargs)
                compute_time = time.time() - start_time
                
                cache[key] = result
                cache_times[key] = time.time()
                
                self.logger.debug(f"Cached result for {func.__name__} (computed in {compute_time:.4f}s)")
                return result
            
            wrapper.cache_clear = lambda: cache.clear() or cache_times.clear()
            wrapper.cache_info = lambda: {'size': len(cache), 'hits': 0, 'misses': 0}  # Simplified
            return wrapper
        
        return decorator
    
    def disk_cache(self, cache_dir: str = None, max_size: int = 1000):
        """Decorator for persistent disk caching."""
        cache_path = Path(cache_dir) if cache_dir else self.cache_dir
        cache_path.mkdir(parents=True, exist_ok=True)
        
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Create cache key and filename
                key = self._create_cache_key(args, kwargs)
                cache_file = cache_path / f"{func.__name__}_{key[:16]}.pkl"
                
                # Try to load from cache
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            cached_data = pickle.load(f)
                        
                        # Check if cache is still valid (simple timestamp check)
                        if time.time() - cache_file.stat().st_mtime < 3600:  # 1 hour TTL
                            self.logger.debug(f"Disk cache hit for {func.__name__}")
                            return cached_data
                    except Exception as e:
                        self.logger.warning(f"Failed to load cache for {func.__name__}: {e}")
                
                # Compute result and save to cache
                result = func(*args, **kwargs)
                
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(result, f)
                    self.logger.debug(f"Saved to disk cache: {func.__name__}")
                except Exception as e:
                    self.logger.warning(f"Failed to save cache for {func.__name__}: {e}")
                
                # Clean up old cache files if needed
                self._cleanup_cache(cache_path, max_size)
                
                return result
            
            return wrapper
        
        return decorator
    
    def profile(self, include_memory: bool = True):
        """Decorator for profiling function performance."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                func_name = f"{func.__module__}.{func.__name__}"
                
                # Memory before
                if include_memory:
                    gc.collect()  # Force garbage collection
                    mem_before = self._get_memory_usage()
                
                # Time execution
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                
                # Memory after
                if include_memory:
                    gc.collect()
                    mem_after = self._get_memory_usage()
                    memory_delta = mem_after - mem_before
                else:
                    memory_delta = 0
                
                # Update statistics
                if func_name not in self.call_stats:
                    self.call_stats[func_name] = {
                        'calls': 0,
                        'total_time': 0.0,
                        'min_time': float('inf'),
                        'max_time': 0.0,
                        'total_memory': 0.0
                    }
                
                stats = self.call_stats[func_name]
                stats['calls'] += 1
                stats['total_time'] += execution_time
                stats['min_time'] = min(stats['min_time'], execution_time)
                stats['max_time'] = max(stats['max_time'], execution_time)
                stats['total_memory'] += memory_delta
                
                # Log performance
                self.logger.debug(f"{func_name}: {execution_time:.4f}s, {memory_delta/1024/1024:.2f}MB")
                
                return result
            
            return wrapper
        
        return decorator
    
    def parallel_map(self, func: Callable, items: List[Any], 
                    use_processes: bool = False, max_workers: int = None) -> List[Any]:
        """Parallel execution of function over list of items."""
        if len(items) == 0:
            return []
        
        if len(items) == 1:
            # Single item, no need for parallelization
            return [func(items[0])]
        
        max_workers = max_workers or min(len(items), 4)
        
        try:
            if use_processes:
                if not self.process_pool:
                    self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
                executor = self.process_pool
            else:
                executor = self.thread_pool
            
            start_time = time.time()
            results = list(executor.map(func, items))
            end_time = time.time()
            
            self.logger.info(f"Parallel execution of {len(items)} items completed in {end_time - start_time:.4f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            # Fallback to sequential execution
            return [func(item) for item in items]
    
    def batch_process(self, func: Callable, items: List[Any], 
                     batch_size: int = 32, progress_callback: Callable = None) -> List[Any]:
        """Process items in batches to optimize memory usage."""
        results = []
        total_batches = (len(items) + batch_size - 1) // batch_size
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = func(batch)
            results.extend(batch_results)
            
            if progress_callback:
                progress_callback(len(results), len(items))
            
            # Optional garbage collection between batches
            if len(results) % (batch_size * 4) == 0:
                gc.collect()
        
        return results
    
    def optimize_jax_function(self, func: Callable, static_argnames: List[str] = None):
        """Optimize JAX function with JIT compilation and caching."""
        if not JAX_AVAILABLE:
            self.logger.warning("JAX not available, returning unoptimized function")
            return func
        
        # Create cache key for this function
        func_key = f"{func.__module__}.{func.__name__}"
        
        if func_key in self.jit_cache:
            return self.jit_cache[func_key]
        
        try:
            # Apply JIT compilation
            jit_func = jax.jit(func, static_argnames=static_argnames or [])
            
            # Warm up the JIT function with dummy data if possible
            try:
                # This would need to be customized based on function signature
                # For now, just return the JIT function
                pass
            except Exception as e:
                self.logger.debug(f"JIT warmup failed for {func_key}: {e}")
            
            self.jit_cache[func_key] = jit_func
            self.logger.info(f"JIT compiled function: {func_key}")
            return jit_func
            
        except Exception as e:
            self.logger.warning(f"JIT compilation failed for {func_key}: {e}")
            return func
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'call_statistics': {},
            'memory_usage': self._get_memory_usage(),
            'cache_statistics': {
                'disk_cache_size': self._get_cache_size(),
                'jit_cache_size': len(self.jit_cache)
            }
        }
        
        # Process call statistics
        for func_name, stats in self.call_stats.items():
            report['call_statistics'][func_name] = {
                'calls': stats['calls'],
                'total_time': stats['total_time'],
                'avg_time': stats['total_time'] / stats['calls'],
                'min_time': stats['min_time'],
                'max_time': stats['max_time'],
                'avg_memory': stats['total_memory'] / stats['calls'] / 1024 / 1024  # MB
            }
        
        return report
    
    def _create_cache_key(self, args: tuple, kwargs: dict) -> str:
        """Create cache key from function arguments."""
        try:
            # Simple hash-based key generation
            key_data = pickle.dumps((args, sorted(kwargs.items())))
            return hashlib.md5(key_data).hexdigest()
        except Exception:
            # Fallback to string representation
            return hashlib.md5(str((args, kwargs)).encode()).hexdigest()
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            # Fallback: use gc module
            return len(gc.get_objects()) * 64  # Rough estimate
    
    def _get_cache_size(self) -> int:
        """Get total size of disk cache in bytes."""
        total_size = 0
        try:
            for cache_file in self.cache_dir.glob('*.pkl'):
                total_size += cache_file.stat().st_size
        except Exception:
            pass
        return total_size
    
    def _cleanup_cache(self, cache_path: Path, max_files: int):
        """Clean up old cache files."""
        try:
            cache_files = list(cache_path.glob('*.pkl'))
            if len(cache_files) > max_files:
                # Sort by modification time and remove oldest
                cache_files.sort(key=lambda f: f.stat().st_mtime)
                for old_file in cache_files[:-max_files]:
                    old_file.unlink()
                    self.logger.debug(f"Removed old cache file: {old_file.name}")
        except Exception as e:
            self.logger.warning(f"Cache cleanup failed: {e}")


class ConcurrentSimulator:
    """Concurrent simulation execution for multiple scenarios."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(4, os.cpu_count() or 1)
        self.logger = get_logger('concurrent_sim')
        self.optimizer = PerformanceOptimizer()
    
    def run_concurrent_simulations(self, 
                                  simulation_configs: List[Dict[str, Any]],
                                  simulator_class: Any,
                                  callback: Callable = None) -> List[Dict[str, Any]]:
        """Run multiple simulations concurrently."""
        
        def run_single_simulation(config):
            try:
                simulator = simulator_class(**config.get('init_params', {}))
                result = simulator.simulate(**config.get('sim_params', {}))
                
                if callback:
                    callback(config, result)
                
                return {
                    'config': config,
                    'result': result,
                    'status': 'success'
                }
            except Exception as e:
                self.logger.error(f"Simulation failed: {e}")
                return {
                    'config': config,
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Use parallel execution
        results = self.optimizer.parallel_map(
            run_single_simulation,
            simulation_configs,
            use_processes=False,  # Use threads for I/O bound simulations
            max_workers=self.max_workers
        )
        
        # Log results summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful
        
        self.logger.info(f"Concurrent simulations completed: {successful} successful, {failed} failed")
        
        return results
    
    def parameter_sweep(self,
                       base_config: Dict[str, Any],
                       parameter_ranges: Dict[str, List[Any]],
                       simulator_class: Any) -> List[Dict[str, Any]]:
        """Run parameter sweep simulations concurrently."""
        
        # Generate all parameter combinations
        import itertools
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        configs = []
        for combination in itertools.product(*param_values):
            config = base_config.copy()
            param_dict = dict(zip(param_names, combination))
            
            # Update configuration with parameter values
            if 'sim_params' not in config:
                config['sim_params'] = {}
            config['sim_params'].update(param_dict)
            
            configs.append(config)
        
        self.logger.info(f"Running parameter sweep with {len(configs)} configurations")
        
        return self.run_concurrent_simulations(configs, simulator_class)


class MemoryManager:
    """Memory management utilities for large-scale simulations."""
    
    def __init__(self):
        self.logger = get_logger('memory')
        self._memory_pools = {}
    
    def create_memory_pool(self, name: str, initial_size: int = 1024*1024):
        """Create a memory pool for efficient allocation."""
        # Simplified memory pool implementation
        self._memory_pools[name] = {
            'size': initial_size,
            'allocated': 0,
            'blocks': []
        }
        self.logger.debug(f"Created memory pool '{name}' with {initial_size} bytes")
    
    def monitor_memory_usage(self, func: Callable):
        """Decorator to monitor memory usage of functions."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            gc.collect()  # Force garbage collection
            
            try:
                import psutil
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss
            except ImportError:
                mem_before = 0
            
            result = func(*args, **kwargs)
            
            try:
                mem_after = process.memory_info().rss
                memory_delta = mem_after - mem_before
                
                if memory_delta > 100 * 1024 * 1024:  # 100MB threshold
                    self.logger.warning(f"High memory usage in {func.__name__}: {memory_delta/1024/1024:.2f}MB")
                else:
                    self.logger.debug(f"Memory usage in {func.__name__}: {memory_delta/1024/1024:.2f}MB")
            except:
                pass
            
            return result
        
        return wrapper
    
    def cleanup_large_objects(self):
        """Force cleanup of large objects and garbage collection."""
        initial_objects = len(gc.get_objects())
        gc.collect()
        final_objects = len(gc.get_objects())
        
        freed_objects = initial_objects - final_objects
        self.logger.debug(f"Garbage collection freed {freed_objects} objects")
        
        return freed_objects


# Global instances
_performance_optimizer = None
_memory_manager = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer

def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager

# Convenience decorators
def memoize(cache_file: str = None, ttl: int = 3600):
    """Convenience decorator for memoization."""
    return get_performance_optimizer().memoize(cache_file, ttl)

def disk_cache(cache_dir: str = None, max_size: int = 1000):
    """Convenience decorator for disk caching."""
    return get_performance_optimizer().disk_cache(cache_dir, max_size)

def profile(include_memory: bool = True):
    """Convenience decorator for profiling."""
    return get_performance_optimizer().profile(include_memory)

def monitor_memory():
    """Convenience decorator for memory monitoring."""
    return get_memory_manager().monitor_memory_usage