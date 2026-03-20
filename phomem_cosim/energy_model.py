"""
energy_model.py — Energy consumption model for photonic-memristive systems.

Optical energy:
    E_optical = sum(|x_i|^2) * power_per_mode * dt
    (simplified: dt = 1 ns assumed, power_per_mode in Watts)

Memristive (PCM programming) energy:
    E_memristive = n_pulses * energy_per_pulse
    (energy_per_pulse typically ~1 pJ for GST PCM)
"""

import numpy as np


class EnergyModel:
    """Models energy consumption in a photonic-memristive co-processor.

    Attributes
    ----------
    dt : float
        Integration time in seconds (default 1 ns = 1e-9 s).
    """

    def __init__(self, dt: float = 1e-9):
        self.dt = dt

    # ------------------------------------------------------------------
    # Individual components
    # ------------------------------------------------------------------

    def compute_optical_energy(
        self, x: np.ndarray, power_per_mode: float = 1e-3
    ) -> float:
        """Compute optical energy dissipated for a given field vector.

        E = dt * power_per_mode * Σ|x_i|²

        Parameters
        ----------
        x : array-like, complex or real
            Optical field amplitudes.
        power_per_mode : float
            Power per unit intensity per mode (Watts). Default 1 mW.

        Returns
        -------
        float : Energy in Joules (always ≥ 0).
        """
        x = np.asarray(x, dtype=complex)
        total_intensity = float(np.sum(np.abs(x) ** 2))
        return self.dt * power_per_mode * total_intensity

    def compute_memristive_energy(
        self, n_pulses: int, energy_per_pulse: float = 1e-12
    ) -> float:
        """Compute energy consumed by PCM programming pulses.

        E = n_pulses * energy_per_pulse

        Parameters
        ----------
        n_pulses : int
            Total number of SET/RESET pulses applied.
        energy_per_pulse : float
            Energy per pulse in Joules. Default 1 pJ (typical for GST PCM).

        Returns
        -------
        float : Energy in Joules (always ≥ 0).
        """
        return int(n_pulses) * float(energy_per_pulse)

    # ------------------------------------------------------------------
    # Combined breakdown
    # ------------------------------------------------------------------

    def total_energy(
        self,
        optical_x: np.ndarray,
        n_pulses: int,
        power_per_mode: float = 1e-3,
        energy_per_pulse: float = 1e-12,
    ) -> dict:
        """Compute total energy with per-component breakdown.

        Parameters
        ----------
        optical_x : array-like
            Optical field vector.
        n_pulses : int
            Number of memristive programming pulses.
        power_per_mode : float
            Optical power per mode (Watts).
        energy_per_pulse : float
            Energy per PCM pulse (Joules).

        Returns
        -------
        dict with keys:
            'optical_J'     : optical energy (J)
            'memristive_J'  : memristive energy (J)
            'total_J'       : sum (J)
        """
        e_opt = self.compute_optical_energy(optical_x, power_per_mode)
        e_mem = self.compute_memristive_energy(n_pulses, energy_per_pulse)
        return {
            "optical_J": e_opt,
            "memristive_J": e_mem,
            "total_J": e_opt + e_mem,
        }
