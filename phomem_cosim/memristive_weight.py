"""
memristive_weight.py — PCM (Phase Change Material) memristive weight state machine.

Models a single GST (Ge2Sb2Te5) PCM cell with:
  - Amorphous state:    high resistance ~1 MΩ  → low conductance ~1 µS
  - Crystalline state:  low  resistance ~1 kΩ  → high conductance ~1 mS
  - Partial states:     crystallization fraction f ∈ [0.0, 1.0]

Conductance interpolation (log-linear):
    G(f) = G_amo * (G_cryst / G_amo)^f
"""

import numpy as np


# Physical constants for GST PCM
_G_AMORPHOUS = 1e-6    # 1 µS  (R = 1 MΩ)
_G_CRYSTALLINE = 1e-3  # 1 mS  (R = 1 kΩ)

# How much each SET pulse increases crystallization fraction
_PULSE_INCREMENT = 0.1  # 10% per pulse


class MemristiveWeight:
    """PCM-based memristive weight with partial crystallization support.

    Parameters
    ----------
    crystallization : float
        Initial crystallization fraction in [0, 1]. Default 0 (amorphous).
    """

    def __init__(self, crystallization: float = 0.0):
        self._validate_fraction(crystallization)
        self._f = float(crystallization)

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def set_pulse(self, n: int = 1):
        """Apply n SET (crystallization) pulses.

        Each pulse increases crystallization fraction by `_PULSE_INCREMENT`,
        clamped to 1.0.

        Parameters
        ----------
        n : int
            Number of SET pulses to apply. Default 1.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        self._f = min(1.0, self._f + n * _PULSE_INCREMENT)

    def reset_pulse(self):
        """Apply a RESET (amorphization) pulse. Returns device to amorphous state."""
        self._f = 0.0

    # ------------------------------------------------------------------
    # Read-out
    # ------------------------------------------------------------------

    def get_conductance(self) -> float:
        """Return current conductance in Siemens (log-linear interpolation).

        G(f) = G_amo * (G_cryst / G_amo)^f
        """
        log_ratio = np.log(_G_CRYSTALLINE / _G_AMORPHOUS)
        return _G_AMORPHOUS * np.exp(self._f * log_ratio)

    def get_state(self) -> str:
        """Return state label: 'amorphous', 'crystalline', or 'partial'."""
        if self._f <= 0.0:
            return "amorphous"
        elif self._f >= 1.0:
            return "crystalline"
        else:
            return "partial"

    @property
    def crystallization_fraction(self) -> float:
        """Current crystallization fraction f ∈ [0.0, 1.0]."""
        return self._f

    @crystallization_fraction.setter
    def crystallization_fraction(self, value: float):
        self._validate_fraction(value)
        self._f = float(value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_fraction(f: float):
        if not (0.0 <= f <= 1.0):
            raise ValueError(f"Crystallization fraction must be in [0, 1], got {f}")

    def __repr__(self) -> str:
        return (
            f"MemristiveWeight(state={self.get_state()!r}, "
            f"f={self._f:.3f}, G={self.get_conductance():.3e} S)"
        )
