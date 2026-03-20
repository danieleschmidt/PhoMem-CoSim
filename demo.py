#!/usr/bin/env python3
"""
demo.py — Photonic-Memristive Co-Simulator demonstration.

Simulates a small 4-mode photonic network:
  1. Initialise memristive weight cells via SET pulses
  2. Build a diagonal conductance weight matrix
  3. Run through an OpticalLayer (MZI mesh)
  4. Compute energy consumption
  5. Compute numerical gradient of a toy loss function
"""

import numpy as np
from phomem_cosim import OpticalLayer, MemristiveWeight, CoDifferentiableSim, EnergyModel


def main():
    print("=" * 60)
    print("  PhoMem-CoSim — Photonic-Memristive Co-Simulator Demo")
    print("=" * 60)

    N_MODES = 4
    np.random.seed(42)

    # ----------------------------------------------------------------
    # 1. Memristive weights — program each cell differently
    # ----------------------------------------------------------------
    print("\n[1] Programming memristive weight cells...")
    n_pulses_list = [0, 3, 6, 10]   # different crystallization levels
    weights = []
    total_pulses = 0

    for i, n in enumerate(n_pulses_list):
        cell = MemristiveWeight()
        cell.set_pulse(n) if n > 0 else None
        total_pulses += n
        weights.append(cell.get_conductance())
        print(
            f"  Cell {i}: {n:2d} pulses → state={cell.get_state()!r:12s} "
            f"G={cell.get_conductance():.3e} S  "
            f"f={cell.crystallization_fraction:.2f}"
        )

    # Normalise conductances to [0, 1] for use as complex weights
    w_array = np.array(weights, dtype=complex)
    w_norm = w_array / w_array.max()

    # ----------------------------------------------------------------
    # 2. Optical layer forward pass
    # ----------------------------------------------------------------
    print("\n[2] Running MZI mesh forward pass...")
    optical = OpticalLayer(N_MODES, seed=7)

    # Input: coherent superposition across all modes
    x_in = np.ones(N_MODES, dtype=complex) / np.sqrt(N_MODES)
    x_in_intensity = np.abs(x_in) ** 2
    print(f"  Input  |x|^2 = {x_in_intensity}")

    # Apply weights (diagonal)
    x_weighted = w_norm * x_in
    x_out = optical.forward(x_weighted)
    x_out_intensity = np.abs(x_out) ** 2
    print(f"  Output |x|^2 = {x_out_intensity.round(4)}")
    print(f"  Output norm = {np.linalg.norm(x_out):.6f}")

    # ----------------------------------------------------------------
    # 3. CoDifferentiableSim forward + gradient
    # ----------------------------------------------------------------
    print("\n[3] CoDifferentiableSim forward pass...")
    sim = CoDifferentiableSim(N_MODES, seed=7)
    y = sim.forward(x_in, np.diag(w_norm))
    y_intensity = np.abs(y) ** 2
    print(f"  Sim output |y|^2 = {y_intensity.round(4)}")

    print("\n[4] Computing numerical gradient (toy MSE loss)...")
    target = np.zeros(N_MODES, dtype=complex)
    target[0] = 1.0  # concentrate energy in mode 0

    def mse_loss(y_pred):
        diff = np.abs(y_pred) - np.abs(target)
        return float(np.sum(diff ** 2))

    grads = sim.numerical_gradient(x_in, np.diag(w_norm), mse_loss)
    print(f"  grad_thetas (first 3): {grads['grad_thetas'][:3].round(6)}")
    print(f"  grad_phis   (first 3): {grads['grad_phis'][:3].round(6)}")
    print(f"  All finite: {np.all(np.isfinite(grads['grad_thetas'])) and np.all(np.isfinite(grads['grad_phis']))}")

    # ----------------------------------------------------------------
    # 4. Energy accounting
    # ----------------------------------------------------------------
    print("\n[5] Energy accounting...")
    energy = EnergyModel(dt=1e-9)
    breakdown = energy.total_energy(x_out, n_pulses=total_pulses)
    print(f"  Optical energy    : {breakdown['optical_J']:.3e} J")
    print(f"  Memristive energy : {breakdown['memristive_J']:.3e} J")
    print(f"  Total energy      : {breakdown['total_J']:.3e} J")

    print("\n✓ Demo complete.")


if __name__ == "__main__":
    main()
