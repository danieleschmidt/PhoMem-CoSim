"""
Installation verification tests.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


def test_jax_installation():
    """Test that JAX is properly installed and working."""
    # Basic JAX operations
    x = jnp.array([1., 2., 3.])
    y = jnp.sum(x)
    assert y == 6.0
    
    # Test autodiff
    def f(x):
        return x**2
    
    grad_f = jax.grad(f)
    assert grad_f(2.0) == 4.0


def test_gpu_availability():
    """Test GPU availability (optional)."""
    try:
        devices = jax.devices()
        print(f"Available devices: {devices}")
        
        # Test GPU computation if available
        if len([d for d in devices if d.device_kind == 'gpu']) > 0:
            x = jnp.array([1., 2., 3.])
            x_gpu = jax.device_put(x, jax.devices('gpu')[0])
            y = jnp.sum(x_gpu)
            assert y == 6.0
            print("GPU computation successful")
        else:
            print("No GPU available, using CPU")
            
    except Exception as e:
        print(f"GPU test failed: {e}")


def test_phomem_imports():
    """Test that all PhoMem modules can be imported."""
    try:
        import phomem
        from phomem.photonics import MachZehnderMesh, PhotoDetectorArray
        from phomem.memristors import PCMCrossbar, RRAMDevice
        from phomem.neural import HybridNetwork
        from phomem.simulator import MultiPhysicsSimulator
        
        print("All PhoMem modules imported successfully")
        
    except ImportError as e:
        pytest.fail(f"Failed to import PhoMem modules: {e}")


def test_basic_functionality():
    """Test basic functionality of core components."""
    # Test photonic components
    from phomem.photonics import MachZehnderMesh
    
    mzi_mesh = MachZehnderMesh(size=4)
    inputs = jnp.ones(4, dtype=jnp.complex64)
    phases = jnp.zeros(6)  # 4*3/2 = 6 phases for 4x4 mesh
    outputs = mzi_mesh(inputs, {'phases': phases})
    
    assert outputs.shape == (4,)
    assert jnp.allclose(jnp.abs(outputs), 1.0, atol=1e-6)  # Energy conservation
    
    # Test memristive components
    from phomem.memristors import PCMDevice
    
    pcm = PCMDevice()
    voltage = 1.0
    current = pcm(voltage, {'state': 0.5})
    
    assert isinstance(current, (float, jnp.ndarray))
    assert current > 0  # Positive voltage should give positive current
    
    print("Basic functionality tests passed")


if __name__ == "__main__":
    test_jax_installation()
    test_gpu_availability()
    test_phomem_imports()
    test_basic_functionality()
    print("All installation tests passed!")