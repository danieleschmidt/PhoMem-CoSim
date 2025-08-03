"""
Installation test module for PhoMem-CoSim.
Run this to verify the installation is working correctly.
"""

import sys
import traceback


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import jax
        import jax.numpy as jnp
        print("✓ JAX imported successfully")
        
        import numpy as np
        print("✓ NumPy imported successfully")
        
        import optax
        print("✓ Optax imported successfully")
        
        import flax
        print("✓ Flax imported successfully")
        
        # Test PhoMem imports
        import phomem
        print("✓ PhoMem package imported successfully")
        
        from phomem.photonics import MachZehnderMesh, PhotoDetectorArray
        print("✓ Photonic components imported successfully")
        
        from phomem.memristors import PCMCrossbar, RRAMDevice
        print("✓ Memristive components imported successfully")
        
        from phomem.neural import HybridNetwork
        print("✓ Neural network components imported successfully")
        
        from phomem.simulator import MultiPhysicsSimulator
        print("✓ Simulator components imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\nTesting basic functionality...")
    
    try:
        import jax.numpy as jnp
        from phomem.photonics import MachZehnderMesh
        from phomem.memristors import PCMDevice
        
        # Test photonic component
        print("  Testing MachZehnder mesh...")
        mzi = MachZehnderMesh(size=4)
        inputs = jnp.ones(4, dtype=jnp.complex64)
        phases = jnp.zeros(6)  # 4*3/2 phases
        outputs = mzi(inputs, {'phases': phases})
        
        assert outputs.shape == (4,), f"Expected shape (4,), got {outputs.shape}"
        # Check that total power is conserved (allowing for small losses)
        input_power = jnp.sum(jnp.abs(inputs)**2)
        output_power = jnp.sum(jnp.abs(outputs)**2)
        assert output_power <= input_power + 1e-6, "Energy conservation failed"
        print("    ✓ MZI mesh working correctly")
        
        # Test memristive component
        print("  Testing PCM device...")
        pcm = PCMDevice()
        voltage = 1.0
        current = pcm(voltage, {'state': 0.5})
        
        assert isinstance(current, (float, jnp.ndarray)), f"Expected number, got {type(current)}"
        assert current > 0, "Expected positive current for positive voltage"
        print("    ✓ PCM device working correctly")
        
        # Test JAX differentiability
        print("  Testing differentiability...")
        import jax
        
        def test_function(x):
            mzi = MachZehnderMesh(size=2)
            inputs = jnp.array([1.0, 0.0], dtype=jnp.complex64)
            phases = jnp.array([x])  # 2*1/2 = 1 phase for 2x2 mesh
            outputs = mzi(inputs, {'phases': phases})
            return jnp.sum(jnp.abs(outputs)**2)
        
        grad_fn = jax.grad(test_function)
        grad_val = grad_fn(0.0)
        print(f"    ✓ Gradient computed: {grad_val}")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_device_availability():
    """Test available compute devices."""
    print("\nTesting device availability...")
    
    try:
        import jax
        
        devices = jax.devices()
        print(f"  Available devices: {len(devices)}")
        
        for i, device in enumerate(devices):
            print(f"    {i}: {device}")
        
        # Test computation on each device
        for device in devices:
            try:
                import jax.numpy as jnp
                x = jax.device_put(jnp.array([1., 2., 3.]), device)
                y = jnp.sum(x)
                print(f"    ✓ Computation successful on {device}")
            except Exception as e:
                print(f"    ✗ Computation failed on {device}: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Device test failed: {e}")
        return False


def test_example_network():
    """Test creating and running a simple hybrid network."""
    print("\nTesting example hybrid network...")
    
    try:
        import jax
        import jax.numpy as jnp
        from phomem.neural import PhotonicLayer, MemristiveLayer, HybridNetwork
        import flax.linen as nn
        
        # Create simple network
        photonic_layer = PhotonicLayer(size=4)
        memristive_layer = MemristiveLayer(input_size=4, output_size=2)
        
        network = HybridNetwork(layers=[photonic_layer, memristive_layer])
        
        # Initialize and test forward pass
        key = jax.random.PRNGKey(42)
        inputs = jnp.ones(4, dtype=jnp.complex64)
        
        params = network.init(key, inputs, training=True)
        outputs = network.apply(params, inputs, training=False)
        
        print(f"    ✓ Network created and tested")
        print(f"    Input shape: {inputs.shape}")
        print(f"    Output shape: {outputs.shape}")
        print(f"    Output values: {outputs}")
        
        return True
        
    except Exception as e:
        print(f"✗ Network test failed: {e}")
        traceback.print_exc()
        return False


def test_multiphysics_simulation():
    """Test multi-physics simulation components."""
    print("\nTesting multi-physics simulation...")
    
    try:
        from phomem.simulator import MultiPhysicsSimulator
        from phomem.simulator.multiphysics import ChipDesign
        
        # Create chip design
        chip = ChipDesign("TestChip")
        chip.set_geometry(
            grid_size=(20, 20, 5),
            physical_size=(100e-6, 100e-6, 20e-6)
        )
        
        # Create simulator
        simulator = MultiPhysicsSimulator(
            optical_solver='BPM',
            thermal_solver='FEM',
            electrical_solver='SPICE',
            coupling='weak'
        )
        
        print("    ✓ Multi-physics simulator created successfully")
        print("    ✓ Chip design configured")
        
        return True
        
    except Exception as e:
        print(f"✗ Multi-physics test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all installation tests."""
    print("PhoMem-CoSim Installation Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_device_availability,
        test_example_network,
        test_multiphysics_simulation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PhoMem-CoSim is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())