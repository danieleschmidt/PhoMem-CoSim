#!/usr/bin/env python3
"""
Minimal test to debug import issues
"""

import sys
from pathlib import Path

# Add the phomem module to the Python path
sys.path.insert(0, str(Path(__file__).parent))

print("Starting minimal import test...")

try:
    print("1. Importing numpy and jax...")
    import numpy as np
    import jax.numpy as jnp
    print("   ✓ numpy and jax imported successfully")

    print("2. Trying to import quantum optimization directly...")
    from phomem.quantum_enhanced_optimization import QuantumAnnealingOptimizer
    print("   ✓ QuantumAnnealingOptimizer imported successfully")

    print("3. Creating simple optimizer...")
    optimizer = QuantumAnnealingOptimizer(
        num_qubits=4,
        num_iterations=5,
        quantum_correction=False
    )
    print("   ✓ Optimizer created successfully")

    print("All tests passed!")

except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()