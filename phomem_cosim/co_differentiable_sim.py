"""
co_differentiable_sim.py — Co-differentiable photonic-memristive simulator.

Combines an OpticalLayer (MZI mesh) with a memristive weight matrix.
Provides forward pass and numerical (finite-difference) gradient w.r.t.
the optical phase parameters.
"""

import numpy as np

from .optical_layer import OpticalLayer


class CoDifferentiableSim:
    """Combines optical MZI mesh with memristive weight matrix.

    The forward pass is:
        y = OpticalLayer.forward(W @ x)

    where W is a diagonal weight matrix built from memristive conductances.

    Parameters
    ----------
    n_modes : int
        Number of optical/memristive modes.
    seed : int or None
        RNG seed passed to OpticalLayer.
    """

    def __init__(self, n_modes: int, seed: int = None):
        self.n_modes = n_modes
        self.optical_layer = OpticalLayer(n_modes, seed=seed)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray, weights_matrix: np.ndarray) -> np.ndarray:
        """Apply memristive weights then optical MZI transform.

        Parameters
        ----------
        x : array-like, shape (n_modes,)
            Complex or real input field.
        weights_matrix : array-like, shape (n_modes, n_modes) or (n_modes,)
            Weight matrix (or diagonal vector) of conductance values.
            Values are normalised to [0, 1] range internally.

        Returns
        -------
        np.ndarray, shape (n_modes,), complex
        """
        x = np.asarray(x, dtype=complex)
        W = np.asarray(weights_matrix, dtype=complex)

        if W.ndim == 1:
            # Treat as diagonal
            if W.shape[0] != self.n_modes:
                raise ValueError(
                    f"Diagonal weight vector must have length {self.n_modes}"
                )
            weighted = W * x
        elif W.ndim == 2:
            if W.shape != (self.n_modes, self.n_modes):
                raise ValueError(
                    f"Weight matrix must be ({self.n_modes}, {self.n_modes})"
                )
            weighted = W @ x
        else:
            raise ValueError("weights_matrix must be 1D or 2D")

        return self.optical_layer.forward(weighted)

    # ------------------------------------------------------------------
    # Numerical gradient
    # ------------------------------------------------------------------

    def numerical_gradient(
        self,
        x: np.ndarray,
        weights_matrix: np.ndarray,
        loss_fn,
        eps: float = 1e-5,
    ) -> dict:
        """Compute finite-difference gradient of loss w.r.t. phase parameters.

        Uses central differences:
            df/dp ≈ (f(p+ε) - f(p-ε)) / (2ε)

        Parameters
        ----------
        x : array-like, shape (n_modes,)
        weights_matrix : array-like
        loss_fn : callable
            Takes complex output array, returns scalar loss.
        eps : float
            Finite-difference step size.

        Returns
        -------
        dict with keys:
            'grad_thetas' : np.ndarray, shape (n_mzi,)
            'grad_phis'   : np.ndarray, shape (n_mzi,)
        """
        thetas_0 = self.optical_layer.thetas.copy()
        phis_0 = self.optical_layer.phis.copy()
        n_mzi = self.optical_layer.n_mzi

        grad_thetas = np.zeros(n_mzi)
        grad_phis = np.zeros(n_mzi)

        # Gradient w.r.t. thetas
        for k in range(n_mzi):
            th_plus = thetas_0.copy()
            th_plus[k] += eps
            self.optical_layer.set_phases(th_plus, phis_0)
            loss_plus = float(np.real(loss_fn(self.forward(x, weights_matrix))))

            th_minus = thetas_0.copy()
            th_minus[k] -= eps
            self.optical_layer.set_phases(th_minus, phis_0)
            loss_minus = float(np.real(loss_fn(self.forward(x, weights_matrix))))

            grad_thetas[k] = (loss_plus - loss_minus) / (2 * eps)

        # Gradient w.r.t. phis
        for k in range(n_mzi):
            ph_plus = phis_0.copy()
            ph_plus[k] += eps
            self.optical_layer.set_phases(thetas_0, ph_plus)
            loss_plus = float(np.real(loss_fn(self.forward(x, weights_matrix))))

            ph_minus = phis_0.copy()
            ph_minus[k] -= eps
            self.optical_layer.set_phases(thetas_0, ph_minus)
            loss_minus = float(np.real(loss_fn(self.forward(x, weights_matrix))))

            grad_phis[k] = (loss_plus - loss_minus) / (2 * eps)

        # Restore original phases
        self.optical_layer.set_phases(thetas_0, phis_0)

        return {"grad_thetas": grad_thetas, "grad_phis": grad_phis}
