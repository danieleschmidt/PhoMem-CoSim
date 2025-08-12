#!/usr/bin/env python3
"""
Generation 1: MAKE IT WORK - Simple Implementation Test

This script runs a simplified version of our novel optimization algorithms
to demonstrate that the basic functionality works correctly.
"""

import logging
import sys
import time
from pathlib import Path
import numpy as np
import jax.numpy as jnp

# Add the phomem module to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from phomem.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(level=logging.INFO)
logger = get_logger(__name__)


def simple_test_function(params):
    """Simple quadratic test function for demonstration."""
    total_loss = 0.0
    for param_array in params.values():
        total_loss += float(jnp.sum(param_array**2))
    return total_loss


def test_quantum_enhanced_optimization():
    """Test quantum-enhanced optimization with simplified parameters."""
    logger.info("Testing Quantum-Enhanced Optimization...")
    
    try:
        from phomem.quantum_enhanced_optimization import QuantumAnnealingOptimizer
        
        # Simple test parameters
        initial_params = {
            'weights': jnp.array(np.random.normal(0, 1, (4, 4))),
            'biases': jnp.array(np.random.normal(0, 0.1, (4,)))
        }
        
        # Create optimizer with minimal settings
        optimizer = QuantumAnnealingOptimizer(
            num_qubits=6,  # Small for fast testing
            num_iterations=20,  # Few iterations for speed
            quantum_correction=False  # Disable for simplicity
        )
        
        # Run optimization
        start_time = time.time()
        result = optimizer.optimize(simple_test_function, initial_params)
        end_time = time.time()
        
        # Check results
        assert result.optimization_result.best_loss < 100, "Loss too high"
        assert result.optimization_result.success, "Optimization failed"
        assert result.quantum_speedup > 0, "Invalid speedup"
        
        logger.info(f"✓ Quantum optimization completed in {end_time - start_time:.2f}s")
        logger.info(f"  Best loss: {result.optimization_result.best_loss:.6f}")
        logger.info(f"  Quantum speedup: {result.quantum_speedup:.2f}x")
        logger.info(f"  Quantum advantage: {result.quantum_advantage:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Quantum optimization test failed: {e}")
        return False


def test_self_healing_optimization():
    """Test self-healing neuromorphic optimization."""
    logger.info("Testing Self-Healing Neuromorphic Optimization...")
    
    try:
        from phomem.self_healing_neuromorphic import SelfHealingNeuromorphicOptimizer
        
        # Simple test parameters
        initial_params = {
            'network_weights': jnp.array(np.random.normal(0, 1, (6, 6))),
            'node_biases': jnp.array(np.random.normal(0, 0.1, (6,)))
        }
        
        # Create optimizer with minimal settings
        optimizer = SelfHealingNeuromorphicOptimizer(
            network_size=(6, 6),  # Small network
            num_iterations=25,    # Few iterations
            enable_memory=False,  # Disable for simplicity
            fault_injection_rate=0.01  # Low fault rate
        )
        
        # Run optimization
        start_time = time.time()
        result = optimizer.optimize(simple_test_function, initial_params)
        end_time = time.time()
        
        # Check results
        assert result.best_loss < 100, "Loss too high"
        assert result.success, "Optimization failed"
        
        final_health = result.hardware_metrics.get('final_health', 0)
        healing_events = result.hardware_metrics.get('healing_events', [])
        
        logger.info(f"✓ Self-healing optimization completed in {end_time - start_time:.2f}s")
        logger.info(f"  Best loss: {result.best_loss:.6f}")
        logger.info(f"  Final system health: {final_health:.3f}")
        logger.info(f"  Healing events: {len(healing_events)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Self-healing optimization test failed: {e}")
        return False


def test_physics_informed_nas():
    """Test physics-informed neural architecture search."""
    logger.info("Testing Physics-Informed Neural Architecture Search...")
    
    try:
        from phomem.physics_informed_nas import PhysicsInformedNAS
        
        # Simple test parameters
        initial_params = {
            'architecture_weights': jnp.array(np.random.normal(0, 1, (5, 5))),
            'component_params': jnp.array(np.random.normal(0, 0.1, (10,)))
        }
        
        # Create optimizer with minimal settings
        optimizer = PhysicsInformedNAS(
            population_size=10,   # Small population
            num_generations=15,   # Few generations
            multi_objective=False,  # Single objective for simplicity
            physics_weight=0.3
        )
        
        # Run optimization
        start_time = time.time()
        result = optimizer.optimize(
            simple_test_function, 
            initial_params,
            target_architecture_size=(4, 4)
        )
        end_time = time.time()
        
        # Check results
        assert result.best_loss < 100, "Loss too high"
        assert result.success, "Optimization failed"
        
        best_arch = result.hardware_metrics.get('best_architecture')
        physics_violations = result.hardware_metrics.get('physics_violations', {})
        
        logger.info(f"✓ PINAS optimization completed in {end_time - start_time:.2f}s")
        logger.info(f"  Best loss: {result.best_loss:.6f}")
        logger.info(f"  Architecture components: {len(best_arch.components) if best_arch else 0}")
        logger.info(f"  Physics violations: {physics_violations.get('total_violations', 0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ PINAS optimization test failed: {e}")
        return False


def test_comprehensive_benchmarking():
    """Test comprehensive benchmarking framework."""
    logger.info("Testing Comprehensive Benchmarking Framework...")
    
    try:
        from phomem.comprehensive_benchmarking import ComprehensiveBenchmarkSuite
        
        # Create minimal benchmark suite
        suite = ComprehensiveBenchmarkSuite(
            output_directory="./test_benchmark_results",
            enable_profiling=False,
            parallel_workers=1
        )
        
        # Create minimal benchmark
        config = suite.create_comprehensive_benchmark(
            benchmark_name="Simple_Test_Benchmark",
            algorithm_categories=['quantum'],  # Only test one category
            test_function_categories=['classical'],
            num_trials=2  # Minimal trials
        )
        
        # Limit to just one algorithm and one function for speed
        config.algorithms = {
            list(config.algorithms.keys())[0]: list(config.algorithms.values())[0]
        }
        config.test_functions = {
            'sphere': config.test_functions['sphere']
        }
        
        # Run benchmark
        start_time = time.time()
        result = suite.run_comprehensive_benchmark(config)
        end_time = time.time()
        
        # Check results
        assert result.execution_time > 0, "Invalid execution time"
        assert len(result.algorithm_results) > 0, "No algorithm results"
        assert result.publication_summary['total_algorithms_tested'] > 0, "No algorithms tested"
        
        logger.info(f"✓ Benchmarking completed in {end_time - start_time:.2f}s")
        logger.info(f"  Algorithms tested: {result.publication_summary['total_algorithms_tested']}")
        logger.info(f"  Test functions: {result.publication_summary['total_test_functions']}")
        logger.info(f"  Best algorithm: {result.publication_summary['overall_best_algorithm']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Benchmarking test failed: {e}")
        return False


def main():
    """Run all Generation 1 tests."""
    
    logger.info("=" * 80)
    logger.info("GENERATION 1: MAKE IT WORK - Simple Implementation Tests")
    logger.info("=" * 80)
    
    test_results = []
    
    # Test 1: Quantum-Enhanced Optimization
    logger.info("\n" + "─" * 60)
    test_results.append(test_quantum_enhanced_optimization())
    
    # Test 2: Self-Healing Neuromorphic Optimization
    logger.info("\n" + "─" * 60)
    test_results.append(test_self_healing_optimization())
    
    # Test 3: Physics-Informed Neural Architecture Search
    logger.info("\n" + "─" * 60)
    test_results.append(test_physics_informed_nas())
    
    # Test 4: Comprehensive Benchmarking
    logger.info("\n" + "─" * 60)
    test_results.append(test_comprehensive_benchmarking())
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("GENERATION 1 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    test_names = [
        "Quantum-Enhanced Optimization",
        "Self-Healing Neuromorphic Optimization", 
        "Physics-Informed Neural Architecture Search",
        "Comprehensive Benchmarking Framework"
    ]
    
    for i, (test_name, passed) in enumerate(zip(test_names, test_results)):
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"  {i+1}. {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED - Generation 1 implementation works!")
        logger.info("Ready to proceed to Generation 2: MAKE IT ROBUST")
        return True
    else:
        logger.error("❌ Some tests failed - need to fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)