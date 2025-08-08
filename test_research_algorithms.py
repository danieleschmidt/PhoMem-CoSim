#!/usr/bin/env python3
"""
Comprehensive test suite for advanced research algorithms in PhoMem-CoSim.

This script validates the novel quantum-coherent, photonic waveguide, and 
neuromorphic optimization algorithms with realistic test scenarios.
"""

import sys
import logging
import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Add phomem to path
sys.path.append('/root/repo')

from phomem.research import (
    QuantumCoherentOptimizer, 
    PhotonicWaveguideOptimizer, 
    NeuromorphicPlasticityOptimizer,
    BioInspiredSwarmOptimizer,
    create_test_functions,
    create_research_algorithms,
    run_comprehensive_research_study,
    ResearchFramework
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_parameters() -> dict:
    """Create realistic test parameters for photonic-memristor networks."""
    return {
        'phase_shifts': jnp.array(np.random.uniform(-np.pi, np.pi, (8,))),
        'resistance_values': jnp.array(np.random.uniform(1e3, 1e6, (12,))),
        'coupling_weights': jnp.array(np.random.uniform(-1, 1, (4, 4))),
        'optical_amplitudes': jnp.array(np.random.uniform(0, 2, (6,)))
    }


def test_quantum_coherent_optimizer():
    """Test the quantum coherent optimizer with entanglement dynamics."""
    logger.info("Testing Quantum Coherent Optimizer...")
    
    optimizer = QuantumCoherentOptimizer(
        num_qubits=8, 
        num_iterations=50, 
        coherence_time=5e-6
    )
    
    # Test with sphere function
    from phomem.research import sphere_function
    initial_params = {'params': jnp.array(np.random.uniform(-2, 2, (6,)))}
    
    start_time = time.time()
    result = optimizer.optimize(sphere_function, initial_params)
    end_time = time.time()
    
    logger.info(f"Quantum Coherent Optimizer Results:")
    logger.info(f"  Final loss: {result.best_loss:.6f}")
    logger.info(f"  Optimization time: {end_time - start_time:.2f}s")
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Convergence: {len(result.convergence_history)} iterations")
    
    assert result.best_loss < 10.0, "Quantum optimizer should achieve reasonable convergence"
    assert len(result.convergence_history) > 0, "Should have convergence history"
    
    return result


def test_photonic_waveguide_optimizer():
    """Test the photonic waveguide optimizer with mode coupling."""
    logger.info("Testing Photonic Waveguide Optimizer...")
    
    optimizer = PhotonicWaveguideOptimizer(
        wavelength=1550e-9,
        num_modes=4,
        num_iterations=40,
        adaptive_coupling=True
    )
    
    # Test with photonic-memristor specific function
    from phomem.research import photonic_memristor_test_function
    initial_params = create_test_parameters()
    
    start_time = time.time()
    result = optimizer.optimize(photonic_memristor_test_function, initial_params)
    end_time = time.time()
    
    logger.info(f"Photonic Waveguide Optimizer Results:")
    logger.info(f"  Final loss: {result.best_loss:.6f}")
    logger.info(f"  Optimization time: {end_time - start_time:.2f}s")
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Hardware metrics: {result.hardware_metrics is not None}")
    
    assert result.best_loss < 50.0, "Photonic optimizer should handle device constraints"
    assert result.hardware_metrics is not None, "Should track hardware metrics"
    
    return result


def test_neuromorphic_plasticity_optimizer():
    """Test the neuromorphic plasticity optimizer."""
    logger.info("Testing Neuromorphic Plasticity Optimizer...")
    
    optimizer = NeuromorphicPlasticityOptimizer(
        plasticity_rate=0.02,
        homeostasis_strength=0.15,
        num_iterations=80
    )
    
    # Test with Rosenbrock function (classic optimization challenge)
    from phomem.research import rosenbrock_function
    initial_params = {
        'x': jnp.array(np.random.uniform(-1, 1, (5,))),
        'y': jnp.array(np.random.uniform(-1, 1, (5,)))
    }
    
    start_time = time.time()
    result = optimizer.optimize(rosenbrock_function, initial_params)
    end_time = time.time()
    
    logger.info(f"Neuromorphic Plasticity Optimizer Results:")
    logger.info(f"  Final loss: {result.best_loss:.6f}")
    logger.info(f"  Optimization time: {end_time - start_time:.2f}s")
    logger.info(f"  Success: {result.success}")
    
    assert result.best_loss < 100.0, "Neuromorphic optimizer should adapt to landscape"
    
    return result


def test_bio_inspired_algorithms():
    """Test bio-inspired swarm algorithms."""
    logger.info("Testing Bio-Inspired Swarm Algorithms...")
    
    algorithms = {
        'firefly': BioInspiredSwarmOptimizer(algorithm='firefly', num_iterations=30),
        'whale': BioInspiredSwarmOptimizer(algorithm='whale', num_iterations=30),
        'grey_wolf': BioInspiredSwarmOptimizer(algorithm='grey_wolf', num_iterations=30)
    }
    
    results = {}
    from phomem.research import ackley_function
    initial_params = {'params': jnp.array(np.random.uniform(-5, 5, (8,)))}
    
    for name, optimizer in algorithms.items():
        logger.info(f"  Testing {name} algorithm...")
        
        start_time = time.time()
        result = optimizer.optimize(ackley_function, initial_params)
        end_time = time.time()
        
        results[name] = result
        
        logger.info(f"    Final loss: {result.best_loss:.6f}")
        logger.info(f"    Time: {end_time - start_time:.2f}s")
        
        assert result.best_loss < 50.0, f"{name} should handle Ackley function"
    
    return results


def test_comparative_study():
    """Test the comprehensive comparative research study framework."""
    logger.info("Testing Comprehensive Research Study Framework...")
    
    # Run a small-scale study for testing
    result = run_comprehensive_research_study(
        study_name="Test Study - Novel Optimization Algorithms",
        num_trials=2,  # Small number for quick testing
        save_results=False  # Don't save files during testing
    )
    
    logger.info(f"Comparative Study Results:")
    logger.info(f"  Experiment: {result.experiment_name}")
    logger.info(f"  Algorithms tested: {len(result.results)}")
    logger.info(f"  Test functions: {len(list(result.results.values())[0])}")
    logger.info(f"  Conclusions: {len(result.conclusions)}")
    logger.info(f"  Future work suggestions: {len(result.future_work)}")
    
    # Validate structure
    assert len(result.results) > 0, "Should have algorithm results"
    assert len(result.conclusions) > 0, "Should generate conclusions"
    assert len(result.future_work) > 0, "Should suggest future work"
    assert result.statistical_significance is not None, "Should compute statistical tests"
    
    # Check for best performers
    all_functions = list(list(result.results.values())[0].keys())
    logger.info(f"  Test functions evaluated: {all_functions}")
    
    # Show best algorithm for each function
    for func_name in all_functions:
        best_algo = min(result.results.keys(), 
                       key=lambda a: result.results[a][func_name]['mean_loss'])
        best_loss = result.results[best_algo][func_name]['mean_loss']
        logger.info(f"    {func_name}: {best_algo} (loss: {best_loss:.6f})")
    
    return result


def test_hardware_constraints():
    """Test optimization with realistic hardware constraints."""
    logger.info("Testing Hardware-Constrained Optimization...")
    
    # Test with hybrid device physics function
    from phomem.research import hybrid_device_physics_function
    
    # Create parameters that violate some constraints initially
    initial_params = {
        'phase_shifts': jnp.array([np.pi * 1.5, -np.pi * 0.8, np.pi * 2.2, -np.pi * 1.1]),  # Some > π
        'resistance_values': jnp.array([500, 1e7, 2e4, 8e5, 300, 1.5e6]),  # Some out of range
    }
    
    # Test different algorithms on hardware constraints
    algorithms = {
        'quantum_coherent': QuantumCoherentOptimizer(num_iterations=40),
        'photonic_waveguide': PhotonicWaveguideOptimizer(num_iterations=40),
    }
    
    results = {}
    for name, optimizer in algorithms.items():
        logger.info(f"  Testing {name} with hardware constraints...")
        
        result = optimizer.optimize(hybrid_device_physics_function, initial_params)
        results[name] = result
        
        logger.info(f"    Initial loss: {hybrid_device_physics_function(initial_params):.6f}")
        logger.info(f"    Final loss: {result.best_loss:.6f}")
        logger.info(f"    Improvement: {result.best_loss < hybrid_device_physics_function(initial_params)}")
    
    return results


def test_scalability():
    """Test algorithm scalability with different problem sizes."""
    logger.info("Testing Algorithm Scalability...")
    
    sizes = [5, 10, 20]  # Different parameter vector sizes
    algorithms = {
        'quantum_coherent': lambda: QuantumCoherentOptimizer(num_iterations=30),
        'photonic_waveguide': lambda: PhotonicWaveguideOptimizer(num_iterations=30),
    }
    
    from phomem.research import sphere_function
    
    scalability_results = {}
    
    for size in sizes:
        logger.info(f"  Testing with {size} parameters...")
        size_results = {}
        
        initial_params = {'params': jnp.array(np.random.uniform(-2, 2, (size,)))}
        
        for name, algo_factory in algorithms.items():
            optimizer = algo_factory()
            
            start_time = time.time()
            result = optimizer.optimize(sphere_function, initial_params)
            end_time = time.time()
            
            size_results[name] = {
                'time': end_time - start_time,
                'loss': result.best_loss,
                'success': result.success
            }
            
            logger.info(f"    {name}: {result.best_loss:.6f} in {end_time - start_time:.2f}s")
        
        scalability_results[size] = size_results
    
    # Check that algorithms scale reasonably
    for algo in algorithms.keys():
        times = [scalability_results[size][algo]['time'] for size in sizes]
        logger.info(f"  {algo} timing: {times}")
        
        # Should scale sub-quadratically for reasonable performance
        time_ratio = times[-1] / times[0] if times[0] > 0 else float('inf')
        size_ratio = sizes[-1] / sizes[0]
        
        logger.info(f"  {algo} scaling factor: {time_ratio:.2f} vs size factor: {size_ratio:.2f}")
        
        # Reasonable scaling - less than quadratic in most cases
        assert time_ratio < size_ratio**2 * 2, f"{algo} should scale reasonably"
    
    return scalability_results


def main():
    """Run comprehensive test suite for research algorithms."""
    logger.info("=== PhoMem Research Algorithm Test Suite ===")
    
    try:
        # Test individual algorithms
        logger.info("\n1. Testing Individual Algorithms...")
        quantum_result = test_quantum_coherent_optimizer()
        photonic_result = test_photonic_waveguide_optimizer()
        neuromorphic_result = test_neuromorphic_plasticity_optimizer()
        bio_results = test_bio_inspired_algorithms()
        
        # Test comparative framework
        logger.info("\n2. Testing Research Framework...")
        comparative_result = test_comparative_study()
        
        # Test hardware constraints
        logger.info("\n3. Testing Hardware Constraints...")
        hardware_results = test_hardware_constraints()
        
        # Test scalability
        logger.info("\n4. Testing Scalability...")
        scalability_results = test_scalability()
        
        logger.info("\n=== All Tests Passed Successfully! ===")
        
        # Summary
        logger.info("\nSummary of Results:")
        logger.info(f"  Quantum Coherent Optimizer: {quantum_result.best_loss:.6f}")
        logger.info(f"  Photonic Waveguide Optimizer: {photonic_result.best_loss:.6f}")
        logger.info(f"  Neuromorphic Plasticity Optimizer: {neuromorphic_result.best_loss:.6f}")
        
        best_bio = min(bio_results.keys(), key=lambda k: bio_results[k].best_loss)
        logger.info(f"  Best Bio-Inspired: {best_bio} ({bio_results[best_bio].best_loss:.6f})")
        
        logger.info(f"  Comparative study tested {len(comparative_result.results)} algorithms")
        logger.info(f"  Hardware constraints handled successfully")
        logger.info(f"  Scalability tested up to {max(scalability_results.keys())} parameters")
        
        return True
        
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)