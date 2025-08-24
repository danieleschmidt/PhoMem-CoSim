#!/usr/bin/env python3
"""Debug the matrix dimension issue."""

import numpy as np

# Debug the matrix dimension issue
photonic_size = 4
memristor_shape = (4, 2)  # 4 inputs, 2 outputs

# Test input
test_input = np.ones(4) * 1e-3  # Shape: (4,)
print(f"Input shape: {test_input.shape}")

# Photonic layer - unitary transformation
phase_matrix = np.random.uniform(0, 2*np.pi, (photonic_size, photonic_size))
print(f"Phase matrix shape: {phase_matrix.shape}")

U = np.exp(1j * phase_matrix)
optical_output = np.abs(U @ test_input.astype(complex))**2
print(f"Optical output shape: {optical_output.shape}")
print(f"Optical output: {optical_output}")

# Optical-to-electrical conversion
electrical_signal = optical_output * 0.8  # Responsivity
print(f"Electrical signal shape: {electrical_signal.shape}")

# Memristor layer - conductance matrix
conductances = np.random.uniform(1e-6, 1e-3, memristor_shape)  # Shape: (4, 2)
print(f"Conductances shape: {conductances.shape}")

# The issue: conductances @ electrical_signal
# conductances: (4, 2)
# electrical_signal: (4,)
# For matrix multiplication A @ B:
# A must be (m, k) and B must be (k, n) to get (m, n)
# But we have (4, 2) @ (4,) which is invalid

# Correct approach: we need to transpose the conductances matrix
# OR reshape the operation to be consistent

# Option 1: Transpose conductances (most common in crossbar operations)
print("\nOption 1: Transpose conductances")
final_output1 = conductances.T @ electrical_signal  # (2, 4) @ (4,) = (2,)
print(f"Final output 1 shape: {final_output1.shape}")
print(f"Final output 1: {final_output1}")

# Option 2: Reshape electrical signal as column vector
print("\nOption 2: Use electrical signal as row vector")
final_output2 = electrical_signal @ conductances  # (4,) @ (4, 2) = (2,)
print(f"Final output 2 shape: {final_output2.shape}")
print(f"Final output 2: {final_output2}")

# They are different! Let's use Option 2 which is more standard for VMM operations
print("\nUsing Option 2 for the corrected implementation")