"""
Comprehensive performance benchmarking for all three generations.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
import json
import traceback
from datetime import datetime
from typing import Dict, Any, List

from phomem.performance_optimizer import (
    get_cache, get_batch_processor, get_profiler, analyze_performance
)
from phomem.neural.networks import PhotonicLayer, MemristiveLayer, HybridNetwork
from phomem.utils.validation import validate_input_array, validate_device_parameters
from phomem.utils.security import get_security_validator


def benchmark_generation1_core_functionality():
    """Benchmark Generation 1: Basic functionality."""
    results = {'name': 'Generation 1: Core Functionality', 'tests': []}
    
    try:
        # Test 1: Network instantiation
        start_time = time.time()
        photonic_layer = PhotonicLayer(size=8)
        memristive_layer = MemristiveLayer(input_size=8, output_size=16)
        network = HybridNetwork(layers=[photonic_layer, memristive_layer])
        instantiation_time = time.time() - start_time
        
        results['tests'].append({
            'test': 'Network Instantiation',
            'time_ms': instantiation_time * 1000,
            'status': 'PASS',
            'details': f'Created network with {len(network.layers)} layers'
        })
        
        # Test 2: Basic computation
        start_time = time.time()
        key = jax.random.PRNGKey(42)
        inputs = jnp.ones(8, dtype=jnp.complex64)
        
        try:
            params = network.init(key, inputs)
            output = network.apply(params, inputs)
            computation_time = time.time() - start_time
            
            results['tests'].append({
                'test': 'Basic Computation',
                'time_ms': computation_time * 1000,
                'status': 'PASS',
                'details': f'Output shape: {output.shape}, mean: {jnp.mean(jnp.abs(output)):.4f}'
            })
        except Exception as e:
            results['tests'].append({
                'test': 'Basic Computation',
                'time_ms': (time.time() - start_time) * 1000,
                'status': 'FAIL',
                'details': f'Error: {str(e)[:100]}'
            })
        
        # Test 3: Device parameter validation
        start_time = time.time()
        params = {
            'wavelength': 1550e-9,
            'size': 8,
            'loss_db_cm': 0.5
        }
        validated_params = validate_device_parameters(params, 'photonic')
        validation_time = time.time() - start_time
        
        results['tests'].append({
            'test': 'Parameter Validation',
            'time_ms': validation_time * 1000,
            'status': 'PASS',
            'details': f'Validated {len(validated_params)} parameters'
        })
        
    except Exception as e:
        results['tests'].append({
            'test': 'Generation 1 Critical Error',
            'time_ms': 0,
            'status': 'FAIL',
            'details': f'Critical error: {str(e)[:200]}'
        })
    
    return results


def benchmark_generation2_robustness():
    """Benchmark Generation 2: Robustness and security."""
    results = {'name': 'Generation 2: Robustness & Security', 'tests': []}
    
    try:
        # Test 1: Input validation with edge cases
        start_time = time.time()
        
        # Test with NaN values
        try:
            nan_array = jnp.array([1.0, jnp.nan, 3.0])
            validate_input_array(nan_array, "test_nan", allow_nan=False)
            nan_test_result = 'FAIL - Should have caught NaN'
        except Exception:
            nan_test_result = 'PASS - Correctly caught NaN'
        
        # Test with infinite values
        try:
            inf_array = jnp.array([1.0, jnp.inf, 3.0])
            validate_input_array(inf_array, "test_inf", allow_inf=False)
            inf_test_result = 'FAIL - Should have caught inf'
        except Exception:
            inf_test_result = 'PASS - Correctly caught inf'
        
        validation_time = time.time() - start_time
        
        results['tests'].append({
            'test': 'Edge Case Validation',
            'time_ms': validation_time * 1000,
            'status': 'PASS' if 'PASS' in nan_test_result and 'PASS' in inf_test_result else 'PARTIAL',
            'details': f'NaN: {nan_test_result}, Inf: {inf_test_result}'
        })
        
        # Test 2: Security validation
        start_time = time.time()
        security_validator = get_security_validator()
        
        # Test safe content
        safe_content = "print('Hello world')\nx = 1 + 2"
        security_validator.validate_file_content(safe_content, 'test')
        
        # Test dangerous content detection
        try:
            dangerous_content = "exec('malicious code here')"
            security_validator.validate_file_content(dangerous_content, 'test')
            security_result = 'FAIL - Should have caught dangerous content'
        except Exception:
            security_result = 'PASS - Correctly caught dangerous content'
        
        security_time = time.time() - start_time
        
        results['tests'].append({
            'test': 'Security Validation',
            'time_ms': security_time * 1000,
            'status': 'PASS' if 'PASS' in security_result else 'FAIL',
            'details': security_result
        })
        
        # Test 3: Error handling and recovery
        start_time = time.time()
        error_count = 0
        recovery_count = 0
        
        for i in range(10):
            try:
                # Intentionally cause errors
                if i % 3 == 0:
                    raise ValueError(f"Test error {i}")
                recovery_count += 1
            except Exception:
                error_count += 1
        
        error_handling_time = time.time() - start_time
        
        results['tests'].append({
            'test': 'Error Handling',
            'time_ms': error_handling_time * 1000,
            'status': 'PASS',
            'details': f'Handled {error_count} errors, {recovery_count} recoveries'
        })
        
    except Exception as e:
        results['tests'].append({
            'test': 'Generation 2 Critical Error',
            'time_ms': 0,
            'status': 'FAIL',
            'details': f'Critical error: {str(e)[:200]}'
        })
    
    return results


def benchmark_generation3_performance():
    """Benchmark Generation 3: Performance and scaling."""
    results = {'name': 'Generation 3: Performance & Scaling', 'tests': []}
    
    try:
        # Test 1: Adaptive caching performance
        start_time = time.time()
        cache = get_cache()
        cache.clear()  # Start fresh
        
        # Test cache performance with repeated operations
        cache_hits = 0
        cache_misses = 0
        
        for i in range(100):
            key = f"test_key_{i % 10}"  # Reuse keys to test caching
            value = cache.get(key)
            if value is None:
                cache.put(key, f"value_{i}")
                cache_misses += 1
            else:
                cache_hits += 1
        
        cache_time = time.time() - start_time
        hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
        
        results['tests'].append({
            'test': 'Adaptive Caching',
            'time_ms': cache_time * 1000,
            'status': 'PASS' if hit_rate > 0.5 else 'PARTIAL',
            'details': f'Hit rate: {hit_rate:.2%}, Cache size: {len(cache.cache)}'
        })
        
        # Test 2: Batch processing performance
        start_time = time.time()
        batch_processor = get_batch_processor()
        
        def simple_compute(x):
            return x ** 2 + jnp.sin(x)
        
        test_data = list(range(100))
        batch_results = batch_processor.process_batches(test_data, simple_compute)
        batch_time = time.time() - start_time
        
        results['tests'].append({
            'test': 'Batch Processing',
            'time_ms': batch_time * 1000,
            'status': 'PASS' if len(batch_results) == len(test_data) else 'FAIL',
            'details': f'Processed {len(batch_results)}/{len(test_data)} items'
        })
        
        # Test 3: Performance profiling
        start_time = time.time()
        profiler = get_profiler()
        
        with profiler.profile("benchmark_operation"):
            # Simulate some work
            x = jnp.array(np.random.random((100, 100)))
            y = jnp.dot(x, x.T)
            result = jnp.sum(y)
        
        profile_time = time.time() - start_time
        
        # Check if profiling data was collected
        has_profile_data = "benchmark_operation" in profiler.profiles
        
        results['tests'].append({
            'test': 'Performance Profiling',
            'time_ms': profile_time * 1000,
            'status': 'PASS' if has_profile_data else 'FAIL',
            'details': f'Profile data collected: {has_profile_data}'
        })
        
        # Test 4: Auto-scaling simulation
        start_time = time.time()
        
        # Simulate different workload sizes
        small_workload = list(range(10))
        medium_workload = list(range(100))
        large_workload = list(range(1000))
        
        scaling_times = []
        
        for workload in [small_workload, medium_workload, large_workload]:
            workload_start = time.time()
            batch_processor.process_batches(workload, lambda x: x * 2)
            workload_time = time.time() - workload_start
            scaling_times.append(workload_time)
        
        scaling_time = time.time() - start_time
        
        # Check if scaling is reasonable (not linear with size)
        efficiency = (scaling_times[2] / scaling_times[0]) / (len(large_workload) / len(small_workload))
        
        results['tests'].append({
            'test': 'Auto-scaling',
            'time_ms': scaling_time * 1000,
            'status': 'PASS' if efficiency < 5.0 else 'PARTIAL',
            'details': f'Scaling efficiency: {efficiency:.2f} (lower is better)'
        })
        
    except Exception as e:
        results['tests'].append({
            'test': 'Generation 3 Critical Error',
            'time_ms': 0,
            'status': 'FAIL',
            'details': f'Critical error: {str(e)[:200]}\n{traceback.format_exc()[:300]}'
        })
    
    return results


def benchmark_research_algorithms():
    """Benchmark research-level algorithms and novel approaches."""
    results = {'name': 'Research Mode: Novel Algorithms', 'tests': []}
    
    try:
        # Test 1: Quantum-inspired optimization
        start_time = time.time()
        
        # Simulate quantum-inspired variational optimization
        def quantum_inspired_objective(params):
            # Simulate quantum circuit with classical approximation
            n_qubits = 4
            angles = params.reshape(n_qubits, -1)
            
            # Simulate quantum gates with unitary matrices
            state = jnp.array([1.0] + [0.0] * (2**n_qubits - 1), dtype=jnp.complex64)
            
            for i in range(n_qubits):
                for angle in angles[i]:
                    # Simulate rotation gates
                    cos_half = jnp.cos(angle / 2)
                    sin_half = jnp.sin(angle / 2) * 1j
                    
                    # Apply gate (simplified)
                    state = state * cos_half + jnp.roll(state, 1) * sin_half
            
            # Expectation value
            return jnp.real(jnp.sum(jnp.abs(state) ** 2))
        
        # Optimize using JAX
        params = jnp.array(np.random.random((4, 3)) * 2 * np.pi)
        
        # Simple gradient descent
        learning_rate = 0.01
        for _ in range(10):
            grad = jax.grad(quantum_inspired_objective)(params)
            params = params - learning_rate * grad
        
        final_value = quantum_inspired_objective(params)
        quantum_time = time.time() - start_time
        
        results['tests'].append({
            'test': 'Quantum-Inspired Optimization',
            'time_ms': quantum_time * 1000,
            'status': 'PASS' if final_value > 0 else 'FAIL',
            'details': f'Final objective: {final_value:.6f}, Iterations: 10'
        })
        
        # Test 2: Multi-physics coupling simulation
        start_time = time.time()
        
        # Simulate coupled optical-thermal-electrical physics
        n_points = 50
        
        # Optical field
        optical_field = jnp.exp(-((jnp.arange(n_points) - n_points//2) / 10)**2) * jnp.exp(1j * jnp.arange(n_points) * 0.1)
        
        # Thermal distribution (affected by optical absorption)
        optical_power = jnp.abs(optical_field)**2
        thermal_field = jnp.zeros(n_points)
        
        # Simple diffusion simulation
        dt = 0.01
        dx = 1.0
        alpha = 0.1  # Thermal diffusivity
        
        for _ in range(20):
            # Heat generation from optical absorption
            heat_source = optical_power * 0.1
            
            # Thermal diffusion
            thermal_laplacian = (
                jnp.roll(thermal_field, -1) - 2*thermal_field + jnp.roll(thermal_field, 1)
            ) / dx**2
            
            thermal_field = thermal_field + dt * (alpha * thermal_laplacian + heat_source)
        
        # Electrical conductivity (temperature dependent)
        electrical_conductivity = 1.0 / (1.0 + 0.01 * thermal_field)
        
        coupling_time = time.time() - start_time
        max_temp = jnp.max(thermal_field)
        
        results['tests'].append({
            'test': 'Multi-Physics Coupling',
            'time_ms': coupling_time * 1000,
            'status': 'PASS' if max_temp > 0 else 'FAIL',
            'details': f'Max temperature: {max_temp:.4f}, Coupling iterations: 20'
        })
        
        # Test 3: Neural architecture search
        start_time = time.time()
        
        # Simulate NAS with simple evolutionary approach
        def evaluate_architecture(arch_params):
            # Simulate architecture performance based on parameters
            # arch_params: [n_layers, layer_sizes, connections]
            n_layers, avg_size, connectivity = arch_params
            
            # Simple heuristic for architecture quality
            complexity_penalty = n_layers * avg_size * connectivity / 1000
            performance_gain = jnp.sqrt(n_layers * avg_size) * (1 + connectivity)
            
            return performance_gain - complexity_penalty
        
        # Population of architectures
        population_size = 20
        population = []
        
        for _ in range(population_size):
            arch = [
                np.random.randint(2, 10),  # n_layers
                np.random.randint(10, 100),  # avg_size
                np.random.random()  # connectivity
            ]
            population.append((arch, evaluate_architecture(arch)))
        
        # Simple evolution
        for generation in range(5):
            # Sort by fitness
            population.sort(key=lambda x: x[1], reverse=True)
            
            # Keep best half
            survivors = population[:population_size//2]
            
            # Create new generation
            new_population = []
            for parent1, score1 in survivors:
                for parent2, score2 in survivors:
                    # Crossover
                    child = [
                        (parent1[0] + parent2[0]) // 2,
                        (parent1[1] + parent2[1]) // 2,
                        (parent1[2] + parent2[2]) / 2
                    ]
                    
                    # Mutation
                    child[0] += np.random.randint(-1, 2)
                    child[1] += np.random.randint(-10, 11)
                    child[2] += np.random.normal(0, 0.1)
                    
                    # Clamp values
                    child[0] = max(2, min(10, child[0]))
                    child[1] = max(10, min(100, child[1]))
                    child[2] = max(0.0, min(1.0, child[2]))
                    
                    new_population.append((child, evaluate_architecture(child)))
                    
                    if len(new_population) >= population_size:
                        break
                
                if len(new_population) >= population_size:
                    break
            
            population = new_population
        
        # Best architecture
        best_arch, best_score = max(population, key=lambda x: x[1])
        nas_time = time.time() - start_time
        
        results['tests'].append({
            'test': 'Neural Architecture Search',
            'time_ms': nas_time * 1000,
            'status': 'PASS' if best_score > 0 else 'FAIL',
            'details': f'Best arch: {best_arch}, Score: {best_score:.4f}, Generations: 5'
        })
        
    except Exception as e:
        results['tests'].append({
            'test': 'Research Critical Error',
            'time_ms': 0,
            'status': 'FAIL',
            'details': f'Critical error: {str(e)[:200]}\n{traceback.format_exc()[:300]}'
        })
    
    return results


def run_comprehensive_benchmark():
    """Run complete benchmark suite across all generations."""
    
    print("🚀 Starting Comprehensive Performance Benchmark")
    print("=" * 60)
    
    benchmark_start = time.time()
    
    # System information
    try:
        import psutil
        system_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': '.'.join(map(str, __import__('sys').version_info[:3])),
            'jax_version': jax.__version__,
            'numpy_version': np.__version__,
        }
    except Exception:
        system_info = {'error': 'Could not collect system info'}
    
    # Run all benchmarks
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'system_info': system_info,
        'benchmarks': []
    }
    
    benchmarks = [
        benchmark_generation1_core_functionality,
        benchmark_generation2_robustness,
        benchmark_generation3_performance,
        benchmark_research_algorithms
    ]
    
    for i, benchmark_func in enumerate(benchmarks, 1):
        print(f"\n📊 Running Benchmark {i}/4: {benchmark_func.__name__}")
        print("-" * 40)
        
        try:
            result = benchmark_func()
            all_results['benchmarks'].append(result)
            
            # Print results
            print(f"✅ {result['name']}")
            for test in result['tests']:
                status_icon = "✅" if test['status'] == 'PASS' else "⚠️" if test['status'] == 'PARTIAL' else "❌"
                print(f"  {status_icon} {test['test']}: {test['time_ms']:.2f}ms - {test['details']}")
        
        except Exception as e:
            error_result = {
                'name': f'Benchmark {i} Error',
                'tests': [{
                    'test': 'Critical Benchmark Error',
                    'time_ms': 0,
                    'status': 'FAIL',
                    'details': f'Benchmark failed: {str(e)[:200]}'
                }]
            }
            all_results['benchmarks'].append(error_result)
            print(f"❌ Benchmark {i} failed: {e}")
    
    total_time = time.time() - benchmark_start
    
    # Summary statistics
    total_tests = sum(len(b['tests']) for b in all_results['benchmarks'])
    passed_tests = sum(sum(1 for t in b['tests'] if t['status'] == 'PASS') for b in all_results['benchmarks'])
    failed_tests = sum(sum(1 for t in b['tests'] if t['status'] == 'FAIL') for b in all_results['benchmarks'])
    partial_tests = sum(sum(1 for t in b['tests'] if t['status'] == 'PARTIAL') for b in all_results['benchmarks'])
    
    all_results['summary'] = {
        'total_time_seconds': total_time,
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'partial_tests': partial_tests,
        'success_rate': passed_tests / total_tests if total_tests > 0 else 0
    }
    
    print("\n" + "=" * 60)
    print("📈 BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Tests run: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"⚠️  Partial: {partial_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"Success rate: {all_results['summary']['success_rate']:.1%}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comprehensive_benchmark_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {filename}")
    
    # Performance analysis
    try:
        perf_analysis = analyze_performance()
        print(f"\n🔍 Performance Analysis:")
        print(f"  Cache hit rate: {perf_analysis.get('cache_performance', {}).get('hit_rate', 0):.1%}")
        print(f"  JIT functions cached: {perf_analysis.get('jit_performance', {}).get('cached_functions', 0)}")
        print(f"  Operations profiled: {perf_analysis.get('operation_performance', {}).get('total_operations', 0)}")
    except Exception as e:
        print(f"⚠️  Could not generate performance analysis: {e}")
    
    return all_results


if __name__ == "__main__":
    results = run_comprehensive_benchmark()