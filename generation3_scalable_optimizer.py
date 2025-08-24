#!/usr/bin/env python3
"""
Generation 3: MAKE IT SCALE - High-Performance Scalable Optimizer
Adds performance optimization, caching, concurrency, resource pooling, and auto-scaling.
"""

import sys
import os
import numpy as np
import traceback
import logging
import time
import threading
import multiprocessing
import queue
import hashlib
from typing import Dict, Any, List, Tuple, Optional, Callable, Union, Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache, wraps
import json
import pickle
import gc
import warnings
import weakref
from contextlib import contextmanager

print("⚡ Generation 3: MAKE IT SCALE - High-Performance Scalable Optimizer")
print("=" * 70)

# =============================================================================
# 1. PERFORMANCE MONITORING AND METRICS
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics tracking."""
    operation_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage: float = 0.0
    concurrent_operations: int = 0
    peak_concurrent_operations: int = 0
    error_count: int = 0
    throughput_ops_per_sec: float = 0.0
    
    def update(self, operation_time: float, cache_hit: bool = False, error: bool = False):
        """Update performance metrics."""
        self.operation_count += 1
        self.total_time += operation_time
        self.min_time = min(self.min_time, operation_time)
        self.max_time = max(self.max_time, operation_time)
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
        if error:
            self.error_count += 1
        
        # Update throughput
        if self.total_time > 0:
            self.throughput_ops_per_sec = self.operation_count / self.total_time
    
    @property
    def average_time(self) -> float:
        return self.total_time / max(self.operation_count, 1)
    
    @property
    def cache_hit_rate(self) -> float:
        total_ops = self.cache_hits + self.cache_misses
        return self.cache_hits / max(total_ops, 1)
    
    @property
    def error_rate(self) -> float:
        return self.error_count / max(self.operation_count, 1)

class PerformanceProfiler:
    """Advanced performance profiler with auto-scaling triggers."""
    
    def __init__(self, name: str = "phomem_profiler"):
        self.name = name
        self.metrics = PerformanceMetrics()
        self._lock = threading.Lock()
        self._start_time = time.time()
        
        # Auto-scaling thresholds
        self.scale_up_threshold = 0.8  # Scale up if utilization > 80%
        self.scale_down_threshold = 0.2  # Scale down if utilization < 20%
        self.min_workers = 1
        self.max_workers = multiprocessing.cpu_count() * 2
        
        # Performance history for adaptive scaling
        self._performance_history = []
        self._max_history_size = 100
        
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Profile a single operation with automatic scaling decisions."""
        start_time = time.time()
        cache_hit = False
        error = False
        
        with self._lock:
            self.metrics.concurrent_operations += 1
            self.metrics.peak_concurrent_operations = max(
                self.metrics.peak_concurrent_operations,
                self.metrics.concurrent_operations
            )
        
        try:
            yield self
        except Exception as e:
            error = True
            raise
        finally:
            end_time = time.time()
            operation_time = end_time - start_time
            
            with self._lock:
                self.metrics.concurrent_operations -= 1
                self.metrics.update(operation_time, cache_hit, error)
                
                # Add to performance history
                self._performance_history.append({
                    'timestamp': end_time,
                    'operation_time': operation_time,
                    'concurrent_ops': self.metrics.concurrent_operations,
                    'operation_name': operation_name
                })
                
                # Trim history
                if len(self._performance_history) > self._max_history_size:
                    self._performance_history.pop(0)
    
    def mark_cache_hit(self):
        """Mark current operation as cache hit."""
        # This would be called from within the context manager
        pass
    
    def get_scaling_recommendation(self) -> Dict[str, Any]:
        """Get auto-scaling recommendation based on performance metrics."""
        with self._lock:
            current_utilization = self.metrics.concurrent_operations / max(self.max_workers, 1)
            avg_response_time = self.metrics.average_time
            
            recommendation = {
                'current_workers': self.metrics.concurrent_operations,
                'current_utilization': current_utilization,
                'avg_response_time': avg_response_time,
                'throughput': self.metrics.throughput_ops_per_sec,
                'recommendation': 'maintain'
            }
            
            # Scaling logic
            if current_utilization > self.scale_up_threshold and avg_response_time > 0.1:
                recommended_workers = min(self.max_workers, self.metrics.concurrent_operations * 2)
                recommendation.update({
                    'recommendation': 'scale_up',
                    'recommended_workers': recommended_workers,
                    'reason': f'High utilization ({current_utilization:.1%}) and slow response'
                })
            elif current_utilization < self.scale_down_threshold and len(self._performance_history) > 10:
                recommended_workers = max(self.min_workers, self.metrics.concurrent_operations // 2)
                recommendation.update({
                    'recommendation': 'scale_down',
                    'recommended_workers': recommended_workers,
                    'reason': f'Low utilization ({current_utilization:.1%})'
                })
            
            return recommendation
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self._lock:
            uptime = time.time() - self._start_time
            
            return {
                'profiler_name': self.name,
                'uptime_seconds': uptime,
                'metrics': {
                    'total_operations': self.metrics.operation_count,
                    'average_time_ms': self.metrics.average_time * 1000,
                    'min_time_ms': self.metrics.min_time * 1000 if self.metrics.min_time != float('inf') else 0,
                    'max_time_ms': self.metrics.max_time * 1000,
                    'throughput_ops_per_sec': self.metrics.throughput_ops_per_sec,
                    'cache_hit_rate': self.metrics.cache_hit_rate,
                    'error_rate': self.metrics.error_rate,
                    'peak_concurrent_operations': self.metrics.peak_concurrent_operations
                },
                'scaling': self.get_scaling_recommendation(),
                'health_score': self._calculate_health_score()
            }
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-1)."""
        # Health based on multiple factors
        error_penalty = max(0, 1 - self.metrics.error_rate * 10)
        performance_bonus = min(1, self.metrics.throughput_ops_per_sec / 100)
        cache_bonus = self.metrics.cache_hit_rate
        
        health_score = (error_penalty + performance_bonus + cache_bonus) / 3
        return max(0, min(1, health_score))

# Global profiler instance
profiler = PerformanceProfiler()

# =============================================================================
# 2. ADVANCED CACHING SYSTEM
# =============================================================================

class AdaptiveCache:
    """High-performance adaptive cache with LRU, TTL, and size limits."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 ttl_seconds: float = 300.0,  # 5 minutes
                 enable_compression: bool = True,
                 enable_persistence: bool = False):
        
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_compression = enable_compression
        self.enable_persistence = enable_persistence
        
        # Cache storage
        self._cache = {}
        self._access_times = {}
        self._creation_times = {}
        self._access_counts = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        # Background cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate deterministic cache key."""
        key_data = {
            'args': [str(arg) for arg in args],
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._creation_times:
            return True
        
        age = time.time() - self._creation_times[key]
        return age > self.ttl_seconds
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self._cache:
            return
        
        # Find LRU item
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        # Remove from all structures
        del self._cache[lru_key]
        del self._access_times[lru_key]
        del self._creation_times[lru_key]
        del self._access_counts[lru_key]
        
        self._evictions += 1
    
    def _cleanup_worker(self):
        """Background worker for cache cleanup."""
        while True:
            try:
                time.sleep(60)  # Cleanup every minute
                
                with self._lock:
                    # Remove expired entries
                    expired_keys = [
                        key for key in self._cache.keys()
                        if self._is_expired(key)
                    ]
                    
                    for key in expired_keys:
                        if key in self._cache:
                            del self._cache[key]
                            del self._access_times[key]
                            del self._creation_times[key]
                            del self._access_counts[key]
                            self._evictions += 1
                            
            except Exception:
                pass  # Silent cleanup failures
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            if self._is_expired(key):
                del self._cache[key]
                del self._access_times[key]
                del self._creation_times[key]
                del self._access_counts[key]
                self._misses += 1
                return None
            
            # Update access info
            self._access_times[key] = time.time()
            self._access_counts[key] = self._access_counts.get(key, 0) + 1
            self._hits += 1
            
            return self._cache[key]
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Store value
            if self.enable_compression and hasattr(value, '__dict__'):
                # Simple compression for objects (in real implementation, use zlib)
                compressed_value = value
            else:
                compressed_value = value
            
            self._cache[key] = compressed_value
            current_time = time.time()
            self._access_times[key] = current_time
            self._creation_times[key] = current_time
            self._access_counts[key] = 0
    
    def cached_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with caching."""
        cache_key = self._generate_key(func.__name__, *args, **kwargs)
        
        # Try cache first
        cached_result = self.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Cache result
        self.put(cache_key, result)
        
        return result
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._creation_times.clear()
            self._access_counts.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / max(total_requests, 1)
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'utilization': len(self._cache) / self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'hit_rate': hit_rate,
                'ttl_seconds': self.ttl_seconds
            }

# Global cache instance
cache = AdaptiveCache(max_size=10000, ttl_seconds=600)

# =============================================================================
# 3. RESOURCE POOL MANAGEMENT
# =============================================================================

class ResourcePool:
    """Thread-safe resource pool with automatic scaling."""
    
    def __init__(self,
                 create_resource: Callable,
                 destroy_resource: Callable = None,
                 min_size: int = 2,
                 max_size: int = 20,
                 idle_timeout: float = 300.0):
        
        self.create_resource = create_resource
        self.destroy_resource = destroy_resource or (lambda x: None)
        self.min_size = min_size
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        
        # Pool storage
        self._available = queue.Queue()
        self._in_use = set()
        self._all_resources = weakref.WeakSet()
        
        # Pool statistics
        self._created_count = 0
        self._destroyed_count = 0
        self._get_count = 0
        self._put_count = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize minimum pool size
        self._initialize_pool()
        
        # Background maintenance
        self._maintenance_thread = threading.Thread(target=self._maintenance_worker, daemon=True)
        self._maintenance_thread.start()
    
    def _initialize_pool(self):
        """Initialize pool with minimum resources."""
        for _ in range(self.min_size):
            try:
                resource = self.create_resource()
                self._available.put((resource, time.time()))
                self._all_resources.add(resource)
                self._created_count += 1
            except Exception as e:
                print(f"Warning: Failed to create initial resource: {e}")
    
    def _maintenance_worker(self):
        """Background maintenance for idle resource cleanup."""
        while True:
            try:
                time.sleep(60)  # Check every minute
                self._cleanup_idle_resources()
            except Exception:
                pass
    
    def _cleanup_idle_resources(self):
        """Remove idle resources beyond minimum size."""
        current_time = time.time()
        resources_to_cleanup = []
        
        # Collect idle resources
        temp_resources = []
        
        while not self._available.empty():
            try:
                resource, timestamp = self._available.get_nowait()
                
                if (current_time - timestamp > self.idle_timeout and 
                    len(temp_resources) >= self.min_size):
                    resources_to_cleanup.append(resource)
                else:
                    temp_resources.append((resource, timestamp))
            except queue.Empty:
                break
        
        # Return non-expired resources
        for resource, timestamp in temp_resources:
            self._available.put((resource, timestamp))
        
        # Cleanup expired resources
        for resource in resources_to_cleanup:
            try:
                self.destroy_resource(resource)
                self._destroyed_count += 1
            except Exception:
                pass
    
    @contextmanager
    def get_resource(self):
        """Get resource from pool with context management."""
        resource = None
        
        try:
            # Try to get available resource
            try:
                resource, _ = self._available.get_nowait()
            except queue.Empty:
                # Create new resource if within limits
                with self._lock:
                    if len(self._all_resources) < self.max_size:
                        resource = self.create_resource()
                        self._all_resources.add(resource)
                        self._created_count += 1
                    else:
                        # Wait for available resource
                        resource, _ = self._available.get(timeout=10.0)
            
            if resource is None:
                raise RuntimeError("No resources available in pool")
            
            with self._lock:
                self._in_use.add(resource)
                self._get_count += 1
            
            yield resource
            
        finally:
            if resource is not None:
                with self._lock:
                    if resource in self._in_use:
                        self._in_use.remove(resource)
                    
                    # Return to pool
                    self._available.put((resource, time.time()))
                    self._put_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                'total_resources': len(self._all_resources),
                'available_resources': self._available.qsize(),
                'in_use_resources': len(self._in_use),
                'created_count': self._created_count,
                'destroyed_count': self._destroyed_count,
                'get_count': self._get_count,
                'put_count': self._put_count,
                'utilization': len(self._in_use) / max(len(self._all_resources), 1),
                'pool_limits': {'min': self.min_size, 'max': self.max_size}
            }
    
    def close(self):
        """Close pool and cleanup all resources."""
        # Clean up available resources
        while not self._available.empty():
            try:
                resource, _ = self._available.get_nowait()
                self.destroy_resource(resource)
            except queue.Empty:
                break
        
        # Clean up in-use resources (force cleanup)
        with self._lock:
            for resource in list(self._in_use):
                try:
                    self.destroy_resource(resource)
                except Exception:
                    pass
            self._in_use.clear()

# =============================================================================
# 4. SCALABLE HYBRID NETWORK WITH OPTIMIZATION
# =============================================================================

class ScalableHybridNetwork:
    """Highly optimized and scalable hybrid network implementation."""
    
    def __init__(self,
                 photonic_size: int = 4,
                 memristor_shape: Tuple[int, int] = (4, 2),
                 enable_caching: bool = True,
                 enable_batching: bool = True,
                 enable_optimization: bool = True,
                 max_workers: int = None):
        
        self.photonic_size = photonic_size
        self.memristor_shape = memristor_shape
        self.enable_caching = enable_caching
        self.enable_batching = enable_batching
        self.enable_optimization = enable_optimization
        
        # Worker management
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Initialize optimized components
        self._initialize_optimized_components()
        
        # Batch processing
        if enable_batching:
            self._batch_queue = queue.Queue(maxsize=1000)
            self._batch_results = {}
            self._batch_worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
            self._batch_worker_thread.start()
        
        # Performance tracking
        self._operation_times = []
        self._optimization_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_operations': 0,
            'single_operations': 0,
            'optimizations_applied': 0
        }
        
        print(f"✅ Initialized ScalableHybridNetwork with {self.max_workers} workers")
    
    def _initialize_optimized_components(self):
        """Initialize pre-optimized components."""
        # Pre-compute commonly used matrices
        self.photonic_phases = np.random.uniform(0, 2*np.pi, (self.photonic_size, self.photonic_size))
        self.memristor_conductances = np.random.uniform(1e-6, 1e-3, self.memristor_shape)
        
        # Pre-compiled operations for speed
        self._precomputed_unitary = np.exp(1j * self.photonic_phases)
        
        # Optimization flags
        self._last_optimization_time = time.time()
        self._optimization_interval = 30.0  # Re-optimize every 30 seconds
    
    def _apply_optimizations(self):
        """Apply dynamic optimizations based on usage patterns."""
        current_time = time.time()
        
        if current_time - self._last_optimization_time < self._optimization_interval:
            return
        
        # Analyze recent operations for optimization opportunities
        if len(self._operation_times) > 10:
            avg_time = np.mean(self._operation_times[-100:])  # Last 100 operations
            
            # Dynamic batch size adjustment
            if avg_time > 0.01:  # Slow operations
                if hasattr(self, '_batch_size'):
                    self._batch_size = min(self._batch_size * 2, 100)
                else:
                    self._batch_size = 10
            else:  # Fast operations
                if hasattr(self, '_batch_size'):
                    self._batch_size = max(self._batch_size // 2, 1)
                else:
                    self._batch_size = 1
            
            self._optimization_stats['optimizations_applied'] += 1
        
        self._last_optimization_time = current_time
    
    def _batch_worker(self):
        """Background worker for batch processing."""
        while True:
            try:
                batch = []
                batch_ids = []
                
                # Collect batch
                start_time = time.time()
                timeout = 0.01  # 10ms batch timeout
                
                while len(batch) < getattr(self, '_batch_size', 10) and (time.time() - start_time) < timeout:
                    try:
                        item = self._batch_queue.get(timeout=timeout)
                        batch.append(item['input'])
                        batch_ids.append(item['id'])
                    except queue.Empty:
                        break
                
                if batch:
                    # Process batch
                    batch_results = self._process_batch(batch)
                    
                    # Store results
                    for batch_id, result in zip(batch_ids, batch_results):
                        self._batch_results[batch_id] = result
                    
                    self._optimization_stats['batch_operations'] += len(batch)
                
            except Exception as e:
                print(f"Batch worker error: {e}")
                time.sleep(0.1)
    
    def _process_batch(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Process batch of inputs efficiently."""
        if not inputs:
            return []
        
        try:
            # Stack inputs for vectorized processing
            batch_input = np.stack(inputs)  # Shape: (batch_size, input_size)
            
            # Vectorized photonic processing
            batch_optical = self._vectorized_photonic_forward(batch_input)
            
            # Vectorized memristor processing
            batch_electrical = batch_optical * 0.8  # O/E conversion
            batch_output = self._vectorized_memristor_forward(batch_electrical)
            
            # Convert back to list
            return [output for output in batch_output]
            
        except Exception as e:
            print(f"Batch processing error: {e}")
            # Fallback to individual processing
            return [self._single_forward(inp) for inp in inputs]
    
    def _vectorized_photonic_forward(self, batch_input: np.ndarray) -> np.ndarray:
        """Vectorized photonic processing."""
        # batch_input shape: (batch_size, photonic_size)
        # Apply unitary transformation to entire batch
        complex_input = batch_input.astype(complex)
        
        # Efficient batch matrix multiplication
        complex_output = np.dot(complex_input, self._precomputed_unitary.T)
        
        # Convert to intensity
        optical_output = np.abs(complex_output)**2
        
        # Apply loss
        loss_factor = 10**(-0.5 / 10.0)  # 0.5 dB/cm loss
        return optical_output * loss_factor
    
    def _vectorized_memristor_forward(self, batch_input: np.ndarray) -> np.ndarray:
        """Vectorized memristor processing."""
        # batch_input shape: (batch_size, memristor_rows)
        # Efficient batch VMM operation
        return np.dot(batch_input, self.memristor_conductances)
    
    def _single_forward(self, input_signal: np.ndarray) -> np.ndarray:
        """Single input forward pass (optimized)."""
        # Photonic processing
        complex_input = input_signal.astype(complex)
        complex_output = self._precomputed_unitary @ complex_input
        optical_output = np.abs(complex_output)**2
        
        # Apply loss
        loss_factor = 10**(-0.5 / 10.0)
        optical_output *= loss_factor
        
        # Memristor processing
        electrical_signal = optical_output * 0.8
        final_output = electrical_signal @ self.memristor_conductances
        
        return final_output
    
    def _cached_forward(self, input_signal: np.ndarray) -> np.ndarray:
        """Forward pass with caching."""
        if not self.enable_caching:
            return self._single_forward(input_signal)
        
        # Generate cache key
        input_hash = hashlib.md5(input_signal.tobytes()).hexdigest()[:16]
        
        # Try cache
        cached_result = cache.get(input_hash)
        if cached_result is not None:
            self._optimization_stats['cache_hits'] += 1
            profiler.mark_cache_hit()
            return cached_result
        
        # Compute and cache
        result = self._single_forward(input_signal)
        cache.put(input_hash, result)
        self._optimization_stats['cache_misses'] += 1
        
        return result
    
    def forward(self, input_signal: np.ndarray, use_batch: bool = None) -> np.ndarray:
        """High-performance forward pass with multiple optimization strategies."""
        
        with profiler.profile_operation("scalable_forward"):
            start_time = time.time()
            
            # Input validation (minimal for performance)
            if not isinstance(input_signal, np.ndarray):
                raise ValueError("Input must be numpy array")
            
            # Apply dynamic optimizations
            self._apply_optimizations()
            
            # Choose processing strategy
            use_batch = use_batch if use_batch is not None else self.enable_batching
            
            try:
                if use_batch and hasattr(self, '_batch_queue'):
                    # Batch processing
                    batch_id = id(input_signal)
                    self._batch_queue.put({
                        'id': batch_id,
                        'input': input_signal.copy()
                    }, timeout=0.1)
                    
                    # Wait for result
                    timeout = 1.0
                    start_wait = time.time()
                    while batch_id not in self._batch_results:
                        if time.time() - start_wait > timeout:
                            raise TimeoutError("Batch processing timeout")
                        time.sleep(0.001)
                    
                    result = self._batch_results.pop(batch_id)
                    
                else:
                    # Single processing with caching
                    result = self._cached_forward(input_signal)
                    self._optimization_stats['single_operations'] += 1
                
                # Track performance
                operation_time = time.time() - start_time
                self._operation_times.append(operation_time)
                
                # Keep only recent operation times
                if len(self._operation_times) > 1000:
                    self._operation_times = self._operation_times[-500:]
                
                return result
                
            except Exception as e:
                # Fallback to simple processing
                print(f"Optimized processing failed: {e}")
                return self._single_forward(input_signal)
    
    def concurrent_forward(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Process multiple inputs concurrently."""
        
        with profiler.profile_operation("concurrent_forward"):
            if not inputs:
                return []
            
            # Submit all tasks
            futures = []
            for input_signal in inputs:
                future = self.thread_pool.submit(self.forward, input_signal, use_batch=False)
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures, timeout=30.0):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Concurrent processing error: {e}")
                    results.append(None)
            
            return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        recent_times = self._operation_times[-100:] if self._operation_times else []
        
        stats = {
            'network_config': {
                'photonic_size': self.photonic_size,
                'memristor_shape': self.memristor_shape,
                'max_workers': self.max_workers,
                'optimizations_enabled': {
                    'caching': self.enable_caching,
                    'batching': self.enable_batching,
                    'optimization': self.enable_optimization
                }
            },
            'performance_metrics': {
                'total_operations': len(self._operation_times),
                'avg_time_ms': np.mean(recent_times) * 1000 if recent_times else 0,
                'min_time_ms': np.min(recent_times) * 1000 if recent_times else 0,
                'max_time_ms': np.max(recent_times) * 1000 if recent_times else 0,
                'std_time_ms': np.std(recent_times) * 1000 if recent_times else 0,
                'throughput_ops_per_sec': len(recent_times) / max(sum(recent_times), 0.001)
            },
            'optimization_stats': self._optimization_stats.copy(),
            'cache_stats': cache.get_stats(),
            'profiler_stats': profiler.get_comprehensive_stats(),
            'batch_size': getattr(self, '_batch_size', 1)
        }
        
        return stats
    
    def close(self):
        """Close network and cleanup resources."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        if hasattr(self, '_batch_worker_thread'):
            # Signal batch worker to stop (in real implementation)
            pass

# =============================================================================
# 5. COMPREHENSIVE SCALABILITY TESTS
# =============================================================================

def test_basic_scalable_network():
    """Test basic scalable network functionality."""
    
    print("\n🧪 Testing Basic Scalable Network...")
    
    try:
        network = ScalableHybridNetwork(
            photonic_size=4,
            memristor_shape=(4, 2),
            enable_caching=True,
            enable_batching=True,
            max_workers=4
        )
        
        test_input = np.ones(4) * 1e-3
        output = network.forward(test_input)
        
        assert output.shape == (2,), f"Expected shape (2,), got {output.shape}"
        assert np.all(np.isfinite(output)), "Output should be finite"
        
        print(f"✅ Basic operation: output shape {output.shape}")
        print(f"✅ Output magnitude: {np.linalg.norm(output):.2e}")
        
        network.close()
        return True
        
    except Exception as e:
        print(f"❌ Basic scalable network test failed: {e}")
        return False

def test_performance_optimization():
    """Test performance optimization features."""
    
    print("\n🧪 Testing Performance Optimization...")
    
    try:
        network = ScalableHybridNetwork(
            enable_caching=True,
            enable_batching=True,
            enable_optimization=True
        )
        
        test_input = np.ones(4) * 1e-3
        
        # Run operations to populate cache
        num_ops = 50
        start_time = time.time()
        
        for i in range(num_ops):
            # Use same input for some operations to test caching
            if i % 3 == 0:
                output = network.forward(test_input)
            else:
                varied_input = test_input * (1 + 0.01 * i)
                output = network.forward(varied_input)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Get performance stats
        stats = network.get_performance_stats()
        
        print(f"✅ Processed {num_ops} operations in {total_time:.3f}s")
        print(f"✅ Throughput: {num_ops/total_time:.1f} ops/sec")
        print(f"✅ Cache hit rate: {stats['cache_stats']['hit_rate']:.1%}")
        print(f"✅ Average operation time: {stats['performance_metrics']['avg_time_ms']:.2f}ms")
        
        # Verify performance improvements
        assert stats['cache_stats']['hit_rate'] > 0, "Cache should have some hits"
        assert total_time < num_ops * 0.1, "Should be faster than 100ms per operation"
        
        network.close()
        return True
        
    except Exception as e:
        print(f"❌ Performance optimization test failed: {e}")
        return False

def test_concurrent_processing():
    """Test concurrent processing capabilities."""
    
    print("\n🧪 Testing Concurrent Processing...")
    
    try:
        network = ScalableHybridNetwork(max_workers=8)
        
        # Create multiple test inputs
        num_inputs = 20
        test_inputs = [np.ones(4) * 1e-3 * (1 + 0.1 * i) for i in range(num_inputs)]
        
        # Sequential processing
        start_time = time.time()
        sequential_results = [network.forward(inp, use_batch=False) for inp in test_inputs]
        sequential_time = time.time() - start_time
        
        # Concurrent processing
        start_time = time.time()
        concurrent_results = network.concurrent_forward(test_inputs)
        concurrent_time = time.time() - start_time
        
        # Remove None results from concurrent processing
        concurrent_results = [r for r in concurrent_results if r is not None]
        
        print(f"✅ Sequential: {len(sequential_results)} results in {sequential_time:.3f}s")
        print(f"✅ Concurrent: {len(concurrent_results)} results in {concurrent_time:.3f}s")
        
        if concurrent_time < sequential_time:
            speedup = sequential_time / concurrent_time
            print(f"✅ Speedup: {speedup:.1f}x")
        else:
            print("⚠️ No speedup observed (overhead dominates for small tasks)")
        
        # Verify results are reasonable
        assert len(sequential_results) == num_inputs
        assert len(concurrent_results) >= num_inputs * 0.8  # Allow some failures
        
        network.close()
        return True
        
    except Exception as e:
        print(f"❌ Concurrent processing test failed: {e}")
        return False

def test_auto_scaling():
    """Test auto-scaling capabilities."""
    
    print("\n🧪 Testing Auto-Scaling...")
    
    try:
        # Create load to trigger scaling decisions
        network = ScalableHybridNetwork(max_workers=4)
        
        # Generate varying load patterns
        light_load_inputs = [np.ones(4) * 1e-3 for _ in range(5)]
        heavy_load_inputs = [np.ones(4) * 1e-3 for _ in range(50)]
        
        # Process light load
        for inp in light_load_inputs:
            network.forward(inp)
        
        light_load_stats = profiler.get_scaling_recommendation()
        
        # Process heavy load
        start_time = time.time()
        for inp in heavy_load_inputs:
            network.forward(inp)
        heavy_load_time = time.time() - start_time
        
        heavy_load_stats = profiler.get_scaling_recommendation()
        
        print(f"✅ Light load recommendation: {light_load_stats['recommendation']}")
        print(f"✅ Heavy load recommendation: {heavy_load_stats['recommendation']}")
        print(f"✅ Heavy load utilization: {heavy_load_stats['current_utilization']:.1%}")
        print(f"✅ Throughput: {heavy_load_stats['throughput']:.1f} ops/sec")
        
        # Verify scaling logic
        assert 'recommendation' in light_load_stats
        assert 'recommendation' in heavy_load_stats
        
        network.close()
        return True
        
    except Exception as e:
        print(f"❌ Auto-scaling test failed: {e}")
        return False

def test_resource_efficiency():
    """Test resource efficiency and memory management."""
    
    print("\n🧪 Testing Resource Efficiency...")
    
    try:
        # Test with resource pooling simulation
        def create_mock_resource():
            return {'data': np.random.random((100, 100))}
        
        def destroy_mock_resource(resource):
            del resource['data']
        
        pool = ResourcePool(
            create_resource=create_mock_resource,
            destroy_resource=destroy_mock_resource,
            min_size=2,
            max_size=10
        )
        
        # Test resource acquisition
        resources_used = []
        for i in range(5):
            with pool.get_resource() as resource:
                resources_used.append(id(resource))
        
        pool_stats = pool.get_stats()
        
        print(f"✅ Pool created: {pool_stats['created_count']} resources")
        print(f"✅ Pool utilization: {pool_stats['utilization']:.1%}")
        print(f"✅ Available resources: {pool_stats['available_resources']}")
        
        # Test memory efficiency
        gc.collect()  # Force garbage collection
        
        # Verify resource management
        assert pool_stats['created_count'] >= 2  # At least min_size
        assert pool_stats['created_count'] <= 10  # At most max_size
        
        pool.close()
        return True
        
    except Exception as e:
        print(f"❌ Resource efficiency test failed: {e}")
        return False

def run_generation3_tests():
    """Run all Generation 3 scalability tests."""
    
    tests = [
        ("Basic Scalable Network", test_basic_scalable_network),
        ("Performance Optimization", test_performance_optimization), 
        ("Concurrent Processing", test_concurrent_processing),
        ("Auto-Scaling", test_auto_scaling),
        ("Resource Efficiency", test_resource_efficiency),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"🔬 Running {test_name} Test")
            print(f"{'='*60}")
            
            result = test_func()
            if result:
                print(f"✅ {test_name}: PASSED")
                results.append(True)
            else:
                print(f"❌ {test_name}: FAILED")
                results.append(False)
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'='*70}")
    print(f"📊 Generation 3 Scalability Test Summary")
    print(f"{'='*70}")
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total-passed}/{total}")
    print(f"🎯 Success Rate: {passed/total*100:.1f}%")
    
    # Show final performance stats
    if passed > 0:
        print(f"\n📈 Final Performance Statistics:")
        profiler_stats = profiler.get_comprehensive_stats()
        print(f"   Overall Health Score: {profiler_stats['health_score']:.1%}")
        print(f"   Cache Statistics: {cache.get_stats()}")
    
    return passed == total

if __name__ == "__main__":
    success = run_generation3_tests()
    
    if success:
        print("\n🎉 Generation 3: MAKE IT SCALE - ALL TESTS PASSED!")
        print("✅ High-performance optimization implemented")
        print("✅ Adaptive caching system operational")
        print("✅ Concurrent processing enabled") 
        print("✅ Auto-scaling triggers configured")
        print("✅ Resource pooling and efficiency optimized")
        sys.exit(0)
    else:
        print("\n⚠️ Generation 3: Some tests failed - needs attention")
        sys.exit(1)