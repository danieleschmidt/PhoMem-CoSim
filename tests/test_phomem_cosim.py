"""
test_phomem_cosim.py — Test suite for the PhoMem-CoSim package.

13 tests covering:
  - OpticalLayer: unitarity, output shape, set_phases
  - MemristiveWeight: initial state, set_pulse, reset_pulse, partial states
  - CoDifferentiableSim: forward shape, numerical gradient finiteness
  - EnergyModel: positive energies, total_energy dict
  - Demo: runs without error
"""

import subprocess
import sys
import numpy as np
import pytest

from phomem_cosim import OpticalLayer, MemristiveWeight, CoDifferentiableSim, EnergyModel


# ======================================================================
# OpticalLayer tests
# ======================================================================

class TestOpticalLayer:

    def setup_method(self):
        self.n = 4
        self.layer = OpticalLayer(self.n, seed=0)
        self.x = np.array([1, 0, 0, 0], dtype=complex)

    def test_forward_preserves_norm(self):
        """Unitary forward pass must preserve |x|."""
        x = np.random.default_rng(1).random(self.n) + 1j * np.random.default_rng(2).random(self.n)
        y = self.layer.forward(x)
        assert np.isclose(np.linalg.norm(x), np.linalg.norm(y), atol=1e-10), (
            f"Norm not preserved: |x|={np.linalg.norm(x):.6f}, |y|={np.linalg.norm(y):.6f}"
        )

    def test_output_shape_matches_input(self):
        """Output shape must equal input shape."""
        y = self.layer.forward(self.x)
        assert y.shape == (self.n,), f"Expected shape ({self.n},), got {y.shape}"

    def test_output_is_complex(self):
        """Output dtype must be complex."""
        y = self.layer.forward(self.x)
        assert np.iscomplexobj(y), "Output should be complex"

    def test_set_phases_updates_transform(self):
        """set_phases should change the output."""
        y_before = self.layer.forward(self.x).copy()

        # Set all thetas to pi/4, phis to pi/3
        new_thetas = np.full(self.layer.n_mzi, np.pi / 4)
        new_phis = np.full(self.layer.n_mzi, np.pi / 3)
        self.layer.set_phases(new_thetas, new_phis)

        y_after = self.layer.forward(self.x)
        assert not np.allclose(y_before, y_after), (
            "Output should change after set_phases"
        )

    def test_set_phases_preserves_norm(self):
        """After set_phases, unitary property must still hold."""
        new_thetas = np.full(self.layer.n_mzi, np.pi / 6)
        new_phis = np.full(self.layer.n_mzi, np.pi / 5)
        self.layer.set_phases(new_thetas, new_phis)
        x = np.ones(self.n, dtype=complex) / np.sqrt(self.n)
        y = self.layer.forward(x)
        assert np.isclose(np.linalg.norm(x), np.linalg.norm(y), atol=1e-10)


# ======================================================================
# MemristiveWeight tests
# ======================================================================

class TestMemristiveWeight:

    def test_starts_amorphous(self):
        """Default state is amorphous."""
        cell = MemristiveWeight()
        assert cell.get_state() == "amorphous"
        assert cell.crystallization_fraction == 0.0

    def test_amorphous_conductance(self):
        """Amorphous conductance should be ~1 µS."""
        cell = MemristiveWeight()
        assert np.isclose(cell.get_conductance(), 1e-6, rtol=0.01)

    def test_set_pulse_increases_conductance(self):
        """SET pulse(s) must increase conductance."""
        cell = MemristiveWeight()
        g0 = cell.get_conductance()
        cell.set_pulse(3)
        g1 = cell.get_conductance()
        assert g1 > g0, f"Conductance should increase: {g0:.3e} → {g1:.3e}"

    def test_set_pulse_reaches_crystalline(self):
        """10 SET pulses (10×0.1 = 1.0) should reach crystalline state."""
        cell = MemristiveWeight()
        cell.set_pulse(10)
        assert cell.get_state() == "crystalline"
        assert np.isclose(cell.get_conductance(), 1e-3, rtol=0.01)

    def test_reset_pulse_returns_amorphous(self):
        """RESET pulse must return device to amorphous state."""
        cell = MemristiveWeight()
        cell.set_pulse(5)
        assert cell.get_state() != "amorphous"
        cell.reset_pulse()
        assert cell.get_state() == "amorphous"
        assert cell.crystallization_fraction == 0.0

    def test_partial_crystallization_state(self):
        """Intermediate pulses should give 'partial' state."""
        cell = MemristiveWeight()
        cell.set_pulse(3)  # f = 0.3
        assert cell.get_state() == "partial"
        assert 0.0 < cell.crystallization_fraction < 1.0

    def test_conductance_bounded(self):
        """Conductance must always be within [G_amo, G_cryst]."""
        cell = MemristiveWeight()
        for n in range(11):
            g = cell.get_conductance()
            assert 1e-6 <= g <= 1e-3, f"Conductance out of bounds: {g:.3e}"
            cell.set_pulse(1)


# ======================================================================
# CoDifferentiableSim tests
# ======================================================================

class TestCoDifferentiableSim:

    def setup_method(self):
        self.n = 4
        self.sim = CoDifferentiableSim(self.n, seed=42)
        self.x = np.ones(self.n, dtype=complex) / np.sqrt(self.n)
        self.W = np.eye(self.n, dtype=complex)

    def test_forward_output_shape(self):
        """Forward pass output must have shape (n_modes,)."""
        y = self.sim.forward(self.x, self.W)
        assert y.shape == (self.n,), f"Expected ({self.n},), got {y.shape}"

    def test_forward_output_is_complex(self):
        """Output must be complex."""
        y = self.sim.forward(self.x, self.W)
        assert np.iscomplexobj(y)

    def test_numerical_gradient_finite(self):
        """Numerical gradient must contain only finite values."""
        def loss(y):
            return np.sum(np.abs(y) ** 2)

        grads = self.sim.numerical_gradient(self.x, self.W, loss)
        assert "grad_thetas" in grads
        assert "grad_phis" in grads
        assert np.all(np.isfinite(grads["grad_thetas"])), "grad_thetas has non-finite values"
        assert np.all(np.isfinite(grads["grad_phis"])), "grad_phis has non-finite values"

    def test_numerical_gradient_shape(self):
        """Gradient arrays must match n_mzi shape."""
        def loss(y):
            return np.sum(np.abs(y) ** 2)

        grads = self.sim.numerical_gradient(self.x, self.W, loss)
        n_mzi = self.sim.optical_layer.n_mzi
        assert grads["grad_thetas"].shape == (n_mzi,)
        assert grads["grad_phis"].shape == (n_mzi,)


# ======================================================================
# EnergyModel tests
# ======================================================================

class TestEnergyModel:

    def setup_method(self):
        self.energy = EnergyModel(dt=1e-9)
        self.x = np.ones(4, dtype=complex)

    def test_optical_energy_positive(self):
        """Optical energy must be > 0 for non-zero input."""
        e = self.energy.compute_optical_energy(self.x)
        assert e > 0, f"Expected positive energy, got {e}"

    def test_optical_energy_zero_input(self):
        """Zero input → zero optical energy."""
        e = self.energy.compute_optical_energy(np.zeros(4, dtype=complex))
        assert e == 0.0

    def test_memristive_energy_positive(self):
        """Memristive energy must be > 0 for n_pulses > 0."""
        e = self.energy.compute_memristive_energy(10)
        assert e > 0, f"Expected positive energy, got {e}"

    def test_memristive_energy_zero_pulses(self):
        """Zero pulses → zero memristive energy."""
        e = self.energy.compute_memristive_energy(0)
        assert e == 0.0

    def test_total_energy_returns_dict(self):
        """total_energy must return dict with required keys."""
        result = self.energy.total_energy(self.x, n_pulses=5)
        assert isinstance(result, dict), "total_energy must return dict"
        assert "optical_J" in result
        assert "memristive_J" in result
        assert "total_J" in result

    def test_total_energy_components_sum(self):
        """total_J must equal optical_J + memristive_J."""
        result = self.energy.total_energy(self.x, n_pulses=5)
        assert np.isclose(result["total_J"], result["optical_J"] + result["memristive_J"])


# ======================================================================
# Demo integration test
# ======================================================================

class TestDemo:

    def test_demo_runs_without_error(self):
        """demo.py must execute without raising any exception."""
        result = subprocess.run(
            [sys.executable, "demo.py"],
            capture_output=True,
            text=True,
            cwd="/tmp/PhoMem-CoSim",
        )
        assert result.returncode == 0, (
            f"demo.py failed with return code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
