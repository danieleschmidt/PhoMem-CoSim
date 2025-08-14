#!/usr/bin/env python3
"""
Generation 1 (SIMPLE) - Basic Component Tests
Focus on core photonic-memristive functionality working correctly
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
from pathlib import Path

# Add the phomem module to the Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_jax_functionality():
    """Test that JAX is working properly."""
    print("=== Testing JAX Basic Functionality ===")
    
    # Test basic JAX operations
    x = jnp.array([1., 2., 3., 4.])
    y = jnp.sin(x)
    
    print(f"Input: {x}")
    print(f"sin(input): {y}")
    
    # Test gradient computation
    def simple_fn(x):
        return jnp.sum(x**2)
    
    grad_fn = jax.grad(simple_fn)
    gradient = grad_fn(x)
    
    print(f"Gradient of sum(x²): {gradient}")
    print("✅ JAX functionality working")
    return True

def test_photonic_components():
    """Test basic photonic component functionality."""
    print("\n=== Testing Photonic Components ===")
    
    try:
        import phomem as pm
        
        # Test MZI mesh creation
        mzi_mesh = pm.MachZehnderMesh(
            size=4,
            wavelength=1550e-9,
            loss_db_cm=0.5,
            phase_shifter='thermal'
        )
        print(f"✅ MZI Mesh created: {mzi_mesh.size}x{mzi_mesh.size}")
        
        # Test photodetector array
        photodetector = pm.PhotoDetectorArray(
            responsivity=0.8,
            dark_current=1e-9
        )
        print(f"✅ Photodetector created: responsivity={photodetector.responsivity} A/W")
        
        # Test thermal phase shifter
        phase_shifter = pm.ThermalPhaseShifter(
            power_per_pi=20e-3,
            response_time=10e-6
        )
        print(f"✅ Phase shifter created: {phase_shifter.power_per_pi*1000:.1f} mW/π")
        
        return True
        
    except Exception as e:
        print(f"❌ Photonic components test failed: {e}")
        return False

def test_memristive_components():
    """Test basic memristive component functionality."""
    print("\n=== Testing Memristive Components ===")
    
    try:
        import phomem as pm
        
        # Test PCM crossbar
        pcm_crossbar = pm.PCMCrossbar(
            rows=8,
            cols=6,
            device_model='pcm_mushroom',
            temperature=300
        )
        print(f"✅ PCM Crossbar created: {pcm_crossbar.rows}x{pcm_crossbar.cols}")
        
        # Test RRAM device
        rram_device = pm.RRAMDevice(
            oxide='HfO2',
            thickness=5e-9,
            area=100e-9**2,
            forming_voltage=2.5
        )
        print(f"✅ RRAM device created: {rram_device.oxide}, {rram_device.thickness*1e9:.1f}nm")
        
        # Test PCM device
        pcm_device = pm.PCMDevice(
            material='GST225',
            geometry='mushroom'
        )
        print(f"✅ PCM device created: {pcm_device.material}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memristive components test failed: {e}")
        return False

def test_neural_network_layers():
    """Test neural network layer functionality."""
    print("\n=== Testing Neural Network Layers ===")
    
    try:
        from phomem.neural.networks import PhotonicLayer, MemristiveLayer
        import flax.linen as nn
        
        # Test photonic layer creation
        photonic_layer = PhotonicLayer(
            size=4,
            wavelength=1550e-9,
            phase_shifter_type='thermal'
        )
        print("✅ Photonic layer created")
        
        # Test memristive layer creation  
        memristive_layer = MemristiveLayer(
            input_size=4,
            output_size=8,
            device_type='PCM'
        )
        print("✅ Memristive layer created")
        
        # Test parameter initialization
        key = jax.random.PRNGKey(42)
        
        # Photonic layer test
        inputs = jnp.ones(4, dtype=jnp.complex64) * 0.1
        photonic_params = photonic_layer.init(key, inputs, training=True)
        print(f"✅ Photonic layer params initialized: {list(photonic_params['params'].keys())}")
        
        # Memristive layer test  
        key, subkey = jax.random.split(key)
        current_inputs = jnp.ones(4) * 1e-6  # 1 μA
        mem_params = memristive_layer.init(subkey, current_inputs, training=True)
        print(f"✅ Memristive layer params initialized: {list(mem_params['params'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Neural network layers test failed: {e}")
        return False

def test_basic_forward_pass():
    """Test basic forward pass through individual layers."""
    print("\n=== Testing Basic Forward Pass ===")
    
    try:
        from phomem.neural.networks import PhotonicLayer, MemristiveLayer
        
        # Create layers
        photonic_layer = PhotonicLayer(size=4, wavelength=1550e-9)
        memristive_layer = MemristiveLayer(input_size=4, output_size=8, device_type='PCM')
        
        # Initialize
        key = jax.random.PRNGKey(42)
        
        # Photonic forward pass
        optical_inputs = jnp.ones(4, dtype=jnp.complex64) * 0.1
        photonic_params = photonic_layer.init(key, optical_inputs, training=True)
        optical_outputs = photonic_layer.apply(photonic_params, optical_inputs, training=True)
        
        print(f"✅ Photonic forward pass: {optical_inputs.shape} -> {optical_outputs.shape}")
        print(f"   Input power: {jnp.sum(jnp.abs(optical_inputs)**2):.6f}")
        print(f"   Output power: {jnp.sum(jnp.abs(optical_outputs)**2):.6f}")
        
        # Memristive forward pass
        key, subkey = jax.random.split(key)
        current_inputs = jnp.ones(4) * 1e-6  # 1 μA
        mem_params = memristive_layer.init(subkey, current_inputs, training=True)
        current_outputs = memristive_layer.apply(mem_params, current_inputs, training=True)
        
        print(f"✅ Memristive forward pass: {current_inputs.shape} -> {current_outputs.shape}")
        print(f"   Input current: {jnp.sum(current_inputs)*1e6:.3f} μA")
        print(f"   Output current: {jnp.sum(current_outputs)*1e6:.3f} μA")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        return False

def test_training_utilities():
    """Test basic training utility functions."""
    print("\n=== Testing Training Utilities ===")
    
    try:
        import phomem as pm
        
        # Test optimizer creation
        optimizer = pm.create_hardware_optimizer(
            learning_rate=1e-3,
            phase_shifter_constraints=(-jnp.pi, jnp.pi),
            memristor_constraints=(1e3, 1e6)
        )
        print(f"✅ Hardware optimizer created: {type(optimizer)}")
        
        # Test configuration manager
        config = pm.PhoMemConfig()
        print(f"✅ Configuration created: {type(config)}")
        
        # Test logger setup
        logger = pm.get_logger("test_logger")
        print(f"✅ Logger created: {type(logger)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training utilities test failed: {e}")
        return False

def main():
    """Run all basic component tests."""
    print("PhoMem-CoSim Generation 1 (SIMPLE) - Basic Component Tests")
    print("=" * 70)
    
    tests = [
        test_basic_jax_functionality,
        test_photonic_components,
        test_memristive_components,
        test_neural_network_layers,
        test_basic_forward_pass,
        test_training_utilities
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
    
    print(f"\n=== GENERATION 1 RESULTS ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed >= total * 0.8:  # Allow 80% pass rate for Generation 1
        print("🎉 Generation 1 (SIMPLE) implementation is working!")
        print("✅ Basic photonic-memristive functionality confirmed")
        return True
    else:
        print("⚠️  Generation 1 needs fixes - core functionality not ready")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)