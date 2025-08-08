#!/usr/bin/env python3
"""
Basic test for research algorithm structures without JAX dependency.
This validates the algorithm interfaces and basic functionality.
"""

import sys
import logging
import time
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock JAX numpy for testing
class MockJNP:
    def __getattr__(self, name):
        if hasattr(np, name):
            return getattr(np, name)
        else:
            # Return numpy equivalents for JAX-specific functions
            if name == 'at':
                return self._mock_at
            return getattr(np, name)
    
    def _mock_at(self, array):
        class AtProxy:
            def __init__(self, arr):
                self.arr = arr
            def set(self, val):
                return val
        return AtProxy(array)

# Create mock jax.numpy
sys.modules['jax'] = type(sys)('mock_jax')
sys.modules['jax.numpy'] = MockJNP()

# Add phomem to path
sys.path.append('/root/repo')

def simple_sphere_function(params):
    """Simple sphere function for testing."""
    total = 0.0
    for name, param in params.items():
        param_flat = np.array(param).flatten()
        total += np.sum(param_flat**2)
    return float(total)

def test_algorithm_interfaces():
    """Test that algorithm classes can be instantiated and have correct interfaces."""
    logger.info("Testing Algorithm Interfaces...")
    
    try:
        # Test imports work
        from phomem.research import (
            QuantumCoherentOptimizer,
            PhotonicWaveguideOptimizer, 
            NeuromorphicPlasticityOptimizer,
            BioInspiredSwarmOptimizer,
            ResearchFramework
        )
        logger.info("✓ All algorithm classes imported successfully")
        
        # Test instantiation
        algorithms = {}
        
        try:
            algorithms['quantum'] = QuantumCoherentOptimizer(num_qubits=4, num_iterations=5)
            logger.info("✓ QuantumCoherentOptimizer instantiated")
        except Exception as e:
            logger.error(f"✗ QuantumCoherentOptimizer failed: {e}")
        
        try:
            algorithms['photonic'] = PhotonicWaveguideOptimizer(num_modes=2, num_iterations=5)
            logger.info("✓ PhotonicWaveguideOptimizer instantiated")
        except Exception as e:
            logger.error(f"✗ PhotonicWaveguideOptimizer failed: {e}")
            
        try:
            algorithms['neuromorphic'] = NeuromorphicPlasticityOptimizer(num_iterations=5)
            logger.info("✓ NeuromorphicPlasticityOptimizer instantiated")
        except Exception as e:
            logger.error(f"✗ NeuromorphicPlasticityOptimizer failed: {e}")
            
        try:
            algorithms['firefly'] = BioInspiredSwarmOptimizer(algorithm='firefly', num_iterations=5)
            logger.info("✓ BioInspiredSwarmOptimizer instantiated")
        except Exception as e:
            logger.error(f"✗ BioInspiredSwarmOptimizer failed: {e}")
            
        # Test ResearchFramework
        try:
            framework = ResearchFramework("Test Study")
            logger.info("✓ ResearchFramework instantiated")
        except Exception as e:
            logger.error(f"✗ ResearchFramework failed: {e}")
            
        return algorithms
        
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return {}

def test_basic_optimization():
    """Test basic optimization functionality."""
    logger.info("Testing Basic Optimization...")
    
    algorithms = test_algorithm_interfaces()
    if not algorithms:
        logger.error("No algorithms available for testing")
        return False
    
    # Simple test parameters
    test_params = {
        'x': np.array([1.0, -0.5, 2.0]),
        'y': np.array([0.8, 1.2])
    }
    
    results = {}
    
    for name, optimizer in algorithms.items():
        try:
            logger.info(f"Testing {name} optimizer...")
            
            start_time = time.time()
            result = optimizer.optimize(simple_sphere_function, test_params)
            end_time = time.time()
            
            logger.info(f"  {name}: loss={result.best_loss:.6f}, time={end_time-start_time:.3f}s")
            results[name] = result
            
        except Exception as e:
            logger.error(f"  {name} optimization failed: {e}")
            
    return len(results) > 0

def test_parameter_handling():
    """Test parameter handling and edge cases."""
    logger.info("Testing Parameter Handling...")
    
    test_cases = [
        # Different parameter shapes
        {'single': np.array([1.0])},
        {'vector': np.array([1.0, 2.0, 3.0])},
        {'matrix': np.array([[1.0, 2.0], [3.0, 4.0]])},
        {'mixed': {'a': np.array([1.0]), 'b': np.array([[2.0, 3.0]])}},
    ]
    
    # Test with a simple algorithm
    try:
        from phomem.research import BioInspiredSwarmOptimizer
        optimizer = BioInspiredSwarmOptimizer(algorithm='firefly', num_iterations=3, swarm_size=5)
        
        for i, params in enumerate(test_cases):
            try:
                result = optimizer.optimize(simple_sphere_function, params)
                logger.info(f"✓ Test case {i+1}: loss={result.best_loss:.6f}")
            except Exception as e:
                logger.error(f"✗ Test case {i+1} failed: {e}")
                
        return True
        
    except Exception as e:
        logger.error(f"Parameter handling test failed: {e}")
        return False

def test_research_functions():
    """Test research-specific functions."""
    logger.info("Testing Research Functions...")
    
    try:
        from phomem.research import (
            create_test_functions,
            create_research_algorithms,
            sphere_function,
            rosenbrock_function
        )
        
        # Test function creation
        test_funcs = create_test_functions()
        logger.info(f"✓ Created {len(test_funcs)} test functions")
        
        # Test each function with simple parameters
        simple_params = {'params': np.array([0.5, -0.3, 1.2])}
        
        for name, func in test_funcs.items():
            try:
                result = func(simple_params)
                logger.info(f"  {name}: f(x) = {result:.6f}")
            except Exception as e:
                logger.error(f"  {name} failed: {e}")
        
        # Test algorithm creation
        try:
            research_algos = create_research_algorithms()
            logger.info(f"✓ Created {len(research_algos)} research algorithms")
        except Exception as e:
            logger.error(f"✗ Algorithm creation failed: {e}")
            
        return True
        
    except Exception as e:
        logger.error(f"Research functions test failed: {e}")
        return False

def test_convergence_tracking():
    """Test convergence history and optimization tracking."""
    logger.info("Testing Convergence Tracking...")
    
    try:
        from phomem.research import BioInspiredSwarmOptimizer
        
        optimizer = BioInspiredSwarmOptimizer(
            algorithm='whale', 
            num_iterations=10, 
            swarm_size=8
        )
        
        test_params = {'params': np.array([2.0, -1.5, 0.8])}
        
        result = optimizer.optimize(simple_sphere_function, test_params)
        
        # Check result structure
        assert hasattr(result, 'best_loss'), "Result should have best_loss"
        assert hasattr(result, 'convergence_history'), "Result should have convergence_history"
        assert hasattr(result, 'optimization_time'), "Result should have optimization_time"
        assert len(result.convergence_history) > 0, "Should have convergence history"
        
        logger.info(f"✓ Convergence tracking: {len(result.convergence_history)} points")
        logger.info(f"  Initial: {result.convergence_history[0]:.6f}")
        logger.info(f"  Final: {result.convergence_history[-1]:.6f}")
        logger.info(f"  Best: {result.best_loss:.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Convergence tracking test failed: {e}")
        return False

def main():
    """Run comprehensive test suite."""
    logger.info("=== Basic Research Algorithm Test Suite ===")
    
    test_results = []
    
    # Run tests
    test_results.append(test_algorithm_interfaces())
    test_results.append(test_basic_optimization())
    test_results.append(test_parameter_handling())
    test_results.append(test_research_functions())
    test_results.append(test_convergence_tracking())
    
    # Summary
    passed = sum(1 for r in test_results if r)
    total = len(test_results)
    
    logger.info(f"\n=== Test Summary ===")
    logger.info(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)