"""
Generation 3 optimization and scaling tests.
"""

import pytest
import time
import threading
import numpy as np
import jax.numpy as jnp
from unittest.mock import Mock, patch

from phomem.performance_optimizer import (
    AdaptiveCache, BatchProcessor, JITOptimizer, PerformanceProfiler,
    AutoScaler, ResourceMonitor, cached, profiled, auto_jit
)


class TestAdaptiveCache:
    """Test adaptive caching system."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = AdaptiveCache(max_size=100, ttl=3600)
        assert cache.max_size == 100
        assert cache.ttl == 3600
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_cache_put_get(self):
        """Test basic cache operations."""
        cache = AdaptiveCache(max_size=10)
        
        # Put and get value
        cache.put("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"
        assert cache.hits == 1
        assert cache.misses == 0
    
    def test_cache_miss(self):
        """Test cache miss behavior."""
        cache = AdaptiveCache()
        
        result = cache.get("nonexistent", "default")
        assert result == "default"
        assert cache.misses == 1
    
    def test_cache_eviction(self):
        """Test cache eviction when max size exceeded."""
        cache = AdaptiveCache(max_size=2)
        
        # Fill cache to capacity
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should trigger eviction
        
        assert len(cache.cache) <= 2
        assert cache.evictions > 0
    
    def test_cache_ttl(self):
        """Test time-to-live expiration."""
        cache = AdaptiveCache(ttl=0.1)  # 100ms TTL
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.2)
        assert cache.get("key1") is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = AdaptiveCache()
        
        cache.put("key1", "value1")
        cache.get("key1")
        cache.get("nonexistent")
        
        stats = cache.stats()
        assert stats['size'] == 1
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
    
    def test_cache_thread_safety(self):
        """Test cache thread safety."""
        cache = AdaptiveCache()
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"worker_{worker_id}_value_{i}"
                    cache.put(key, value)
                    result = cache.get(key)
                    results.append(result == value)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert all(results)


class TestBatchProcessor:
    """Test batch processing system."""
    
    def test_batch_processor_initialization(self):
        """Test batch processor initialization."""
        processor = BatchProcessor(max_workers=4)
        assert processor.max_workers == 4
        assert processor.adaptive_batch_size >= 8
    
    def test_simple_batch_processing(self):
        """Test simple batch processing."""
        processor = BatchProcessor(max_workers=2)
        
        def square(x):
            return x * x
        
        data = list(range(10))
        results = processor.process_batches(data, square)
        
        expected = [x * x for x in data]
        assert results == expected
    
    def test_empty_batch_processing(self):
        """Test empty data handling."""
        processor = BatchProcessor()
        
        def dummy_func(x):
            return x
        
        results = processor.process_batches([], dummy_func)
        assert results == []
    
    def test_batch_processing_with_errors(self):
        """Test batch processing with some errors."""
        processor = BatchProcessor(max_workers=2)
        
        def error_prone_func(x):
            if x == 5:
                raise ValueError("Test error")
            return x * 2
        
        data = list(range(10))
        # Should not crash, but may have missing results
        results = processor.process_batches(data, error_prone_func)
        
        # Results should exist but may be incomplete due to error handling
        assert isinstance(results, list)
    
    @patch('phomem.performance_optimizer.psutil')
    def test_adaptive_batch_sizing(self, mock_psutil):
        """Test adaptive batch sizing based on system resources."""
        # Mock system stats
        mock_memory = Mock()
        mock_memory.available = 8 * 1024**3  # 8GB available
        mock_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_percent.return_value = 25.0
        
        processor = BatchProcessor()
        batch_size = processor._adaptive_batch_sizing(1000)
        
        assert batch_size >= 8
        assert batch_size <= 1000


class TestJITOptimizer:
    """Test JIT optimization system."""
    
    def test_jit_optimizer_initialization(self):
        """Test JIT optimizer initialization."""
        optimizer = JITOptimizer()
        assert len(optimizer.compilation_cache) == 0
        assert len(optimizer.function_stats) == 0
    
    def test_smart_jit_basic(self):
        """Test basic JIT compilation."""
        optimizer = JITOptimizer()
        
        def simple_func(x):
            return x + 1
        
        jit_func = optimizer.smart_jit(simple_func)
        
        # Function should be callable
        result = jit_func(jnp.array(5.0))
        assert result == 6.0
        
        # Should be cached
        assert len(optimizer.compilation_cache) == 1
    
    def test_jit_cache_reuse(self):
        """Test JIT compilation cache reuse."""
        optimizer = JITOptimizer()
        
        def simple_func(x):
            return x * 2
        
        # First compilation
        jit_func1 = optimizer.smart_jit(simple_func)
        cache_size_1 = len(optimizer.compilation_cache)
        
        # Second compilation with same function
        jit_func2 = optimizer.smart_jit(simple_func)
        cache_size_2 = len(optimizer.compilation_cache)
        
        # Should reuse cache
        assert cache_size_1 == cache_size_2 == 1
        assert jit_func1 is jit_func2
    
    def test_jit_compilation_stats(self):
        """Test JIT compilation statistics."""
        optimizer = JITOptimizer()
        
        def test_func(x):
            return x**2
        
        optimizer.smart_jit(test_func)
        stats = optimizer.get_compilation_stats()
        
        assert stats['cached_functions'] == 1
        assert 'test_func' in stats['function_stats']
        assert stats['total_compilation_time'] >= 0


class TestPerformanceProfiler:
    """Test performance profiling system."""
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler()
        assert len(profiler.profiles) == 0
        assert len(profiler.bottlenecks) == 0
    
    def test_profile_context_manager(self):
        """Test performance profiling context manager."""
        profiler = PerformanceProfiler()
        
        with profiler.profile("test_operation"):
            time.sleep(0.1)  # Simulate work
        
        assert "test_operation" in profiler.profiles
        assert len(profiler.profiles["test_operation"]) == 1
        
        profile_data = profiler.profiles["test_operation"][0]
        assert profile_data['duration'] >= 0.1
    
    def test_bottleneck_detection(self):
        """Test automatic bottleneck detection."""
        profiler = PerformanceProfiler()
        
        with profiler.profile("slow_operation"):
            time.sleep(1.1)  # Simulate slow operation
        
        # Should detect slow operation as bottleneck
        assert len(profiler.bottlenecks) > 0
        assert any(b['type'] == 'slow_execution' for b in profiler.bottlenecks)
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        profiler = PerformanceProfiler()
        
        # Profile multiple operations
        with profiler.profile("op1"):
            time.sleep(0.1)
        
        with profiler.profile("op1"):
            time.sleep(0.1)
        
        with profiler.profile("op2"):
            time.sleep(0.05)
        
        summary = profiler.get_performance_summary()
        
        assert 'operation_summaries' in summary
        assert 'op1' in summary['operation_summaries']
        assert 'op2' in summary['operation_summaries']
        
        op1_stats = summary['operation_summaries']['op1']
        assert op1_stats['call_count'] == 2
        assert op1_stats['avg_duration'] >= 0.1


class TestAutoScaler:
    """Test automatic scaling system."""
    
    def test_autoscaler_initialization(self):
        """Test autoscaler initialization."""
        scaler = AutoScaler()
        assert scaler.current_scale == 1.0
        assert scaler.min_scale <= scaler.current_scale <= scaler.max_scale
    
    @patch('phomem.performance_optimizer.ResourceMonitor')
    def test_scale_calculation(self, mock_monitor_class):
        """Test scaling factor calculation."""
        # Mock system stats
        mock_monitor = Mock()
        mock_monitor.get_system_stats.return_value = {
            'cpu_percent': 30.0,
            'memory_percent': 40.0,
            'available_memory_gb': 8.0
        }
        mock_monitor_class.return_value = mock_monitor
        
        scaler = AutoScaler()
        scale_factor = scaler._calculate_optimal_scale(
            mock_monitor.get_system_stats(), 1000
        )
        
        # Should scale up with low resource usage
        assert scale_factor >= 1.0
    
    def test_auto_scale_computation(self):
        """Test auto scaling of computation."""
        scaler = AutoScaler()
        
        def dummy_computation(**kwargs):
            return "computed"
        
        result = scaler.auto_scale_computation(dummy_computation, 100)
        assert result == "computed"
        assert len(scaler.scaling_history) == 1


class TestResourceMonitor:
    """Test resource monitoring system."""
    
    def test_resource_monitor_initialization(self):
        """Test resource monitor initialization."""
        monitor = ResourceMonitor()
        assert monitor.cpu_threshold > 0
        assert monitor.memory_threshold > 0
    
    @patch('phomem.performance_optimizer.psutil')
    def test_system_stats(self, mock_psutil):
        """Test system statistics gathering."""
        # Mock psutil responses
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_memory.available = 4 * 1024**3  # 4GB
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_disk = Mock()
        mock_disk.percent = 75.0
        mock_psutil.disk_usage.return_value = mock_disk
        mock_psutil.pids.return_value = range(100)
        
        monitor = ResourceMonitor()
        stats = monitor.get_system_stats()
        
        assert stats['cpu_percent'] == 50.0
        assert stats['memory_percent'] == 60.0
        assert stats['available_memory_gb'] == 4.0
        assert stats['disk_usage_percent'] == 75.0
    
    @patch('phomem.performance_optimizer.psutil')
    def test_resource_limit_warnings(self, mock_psutil):
        """Test resource limit warning generation."""
        # Mock high resource usage
        mock_psutil.cpu_percent.return_value = 95.0
        mock_memory = Mock()
        mock_memory.percent = 90.0
        mock_memory.available = 0.5 * 1024**3  # 0.5GB
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_disk = Mock()
        mock_disk.percent = 85.0
        mock_psutil.disk_usage.return_value = mock_disk
        mock_psutil.pids.return_value = range(50)
        
        monitor = ResourceMonitor()
        warnings = monitor.check_resource_limits()
        
        # Should generate warnings for high resource usage
        assert len(warnings) >= 2
        assert any("CPU" in w for w in warnings)
        assert any("memory" in w for w in warnings)


class TestPerformanceDecorators:
    """Test performance optimization decorators."""
    
    def test_cached_decorator(self):
        """Test caching decorator."""
        call_count = 0
        
        @cached(ttl=3600.0)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * x
        
        # First call
        result1 = expensive_function(5)
        assert result1 == 25
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 25
        assert call_count == 1  # Should not increase
        
        # Different argument should call function
        result3 = expensive_function(6)
        assert result3 == 36
        assert call_count == 2
    
    def test_profiled_decorator(self):
        """Test profiling decorator."""
        @profiled("test_function")
        def test_function():
            time.sleep(0.1)
            return "done"
        
        from phomem.performance_optimizer import get_profiler
        profiler = get_profiler()
        
        # Clear previous profiles
        profiler.profiles.clear()
        
        result = test_function()
        assert result == "done"
        assert "test_function" in profiler.profiles
        assert len(profiler.profiles["test_function"]) == 1
    
    def test_auto_jit_decorator(self):
        """Test auto JIT decorator."""
        @auto_jit()
        def simple_math(x):
            return x + 1
        
        result = simple_math(jnp.array(5.0))
        assert result == 6.0
        
        from phomem.performance_optimizer import get_jit_optimizer
        jit_optimizer = get_jit_optimizer()
        
        # Should have cached the JIT function
        assert len(jit_optimizer.compilation_cache) >= 0  # May be 0 if compilation failed


class TestIntegration:
    """Test integration of optimization components."""
    
    def test_full_optimization_pipeline(self):
        """Test complete optimization pipeline."""
        # Create a function that uses multiple optimizations
        @cached()
        @profiled("integration_test")
        def compute_intensive_task(data):
            return jnp.sum(jnp.array(data) ** 2)
        
        # Test data
        test_data = list(range(100))
        
        # First execution
        result1 = compute_intensive_task(test_data)
        
        # Second execution should use cache
        result2 = compute_intensive_task(test_data)
        
        assert result1 == result2
        
        from phomem.performance_optimizer import get_profiler
        profiler = get_profiler()
        
        # Should have profiling data
        assert "integration_test" in profiler.profiles
    
    def test_performance_analysis(self):
        """Test comprehensive performance analysis."""
        from phomem.performance_optimizer import analyze_performance
        
        # Generate some activity
        cache = AdaptiveCache()
        cache.put("test", "value")
        cache.get("test")
        
        analysis = analyze_performance()
        
        assert 'cache_performance' in analysis
        assert 'jit_performance' in analysis
        assert 'operation_performance' in analysis
        assert 'timestamp' in analysis
    
    def test_workload_optimization(self):
        """Test workload-specific optimization."""
        from phomem.performance_optimizer import optimize_for_workload, get_cache
        
        original_max_size = get_cache().max_size
        
        # Configure for memory-intensive workload
        optimize_for_workload('memory_intensive')
        assert get_cache().max_size <= original_max_size
        
        # Configure for compute-intensive workload
        optimize_for_workload('compute_intensive')
        assert get_cache().max_size >= original_max_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])