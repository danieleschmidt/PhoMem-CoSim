"""
optical_layer.py — MZI mesh optical layer using Clements decomposition (simplified).

The unitary is parameterized by phase shifts (theta, phi) arranged in a
triangular mesh. Each MZI is a 2x2 unitary block:

    U_ij = [[e^{i*phi} * cos(theta/2), -sin(theta/2)],
             [e^{i*phi} * sin(theta/2),  cos(theta/2)]]

The full N-mode unitary is built by composing all MZI blocks in Clements order.
"""

import numpy as np


class OpticalLayer:
    """Mach-Zehnder Interferometer mesh implementing an N-mode unitary transform.

    Parameters
    ----------
    n_modes : int
        Number of optical modes (ports).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, n_modes: int, seed: int = None):
        self.n_modes = n_modes
        rng = np.random.default_rng(seed)

        # Number of MZIs in a Clements mesh: n*(n-1)//2 per diagonal,
        # arranged in n layers. We use the full triangular count.
        self.n_mzi = n_modes * (n_modes - 1) // 2

        # Phase parameters: theta in [0, pi], phi in [0, 2*pi)
        thetas = rng.uniform(0, np.pi, self.n_mzi)
        phis = rng.uniform(0, 2 * np.pi, self.n_mzi)
        self.thetas = thetas.astype(np.float64)
        self.phis = phis.astype(np.float64)

        # Pre-compute MZI placement indices (Clements column pairs)
        self._mzi_pairs = self._build_pairs()

    def _build_pairs(self):
        """Build list of (col_i, col_j) index pairs in Clements column order."""
        n = self.n_modes
        pairs = []
        for col in range(n - 1):
            start = col % 2  # alternate even/odd columns
            for row in range(start, n - 1, 2):
                pairs.append((row, row + 1))
                if len(pairs) == self.n_mzi:
                    return pairs
        return pairs

    def _mzi_matrix(self, theta: float, phi: float) -> np.ndarray:
        """2x2 MZI transfer matrix."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        ep = np.exp(1j * phi)
        return np.array([[ep * c, -s],
                         [ep * s,  c]], dtype=complex)

    def _build_unitary(self) -> np.ndarray:
        """Compose full N-mode unitary from current phase parameters."""
        U = np.eye(self.n_modes, dtype=complex)
        for k, (i, j) in enumerate(self._mzi_pairs):
            M = np.eye(self.n_modes, dtype=complex)
            mzi = self._mzi_matrix(self.thetas[k], self.phis[k])
            M[np.ix_([i, j], [i, j])] = mzi
            U = M @ U
        return U

    def set_phases(self, thetas: np.ndarray, phis: np.ndarray):
        """Update phase parameters.

        Parameters
        ----------
        thetas : array-like, shape (n_mzi,)
        phis   : array-like, shape (n_mzi,)
        """
        thetas = np.asarray(thetas, dtype=np.float64)
        phis = np.asarray(phis, dtype=np.float64)
        if thetas.shape != (self.n_mzi,) or phis.shape != (self.n_mzi,):
            raise ValueError(
                f"Expected arrays of length {self.n_mzi}, "
                f"got {thetas.shape} and {phis.shape}"
            )
        self.thetas = thetas.copy()
        self.phis = phis.copy()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the MZI mesh unitary to input vector x.

        Parameters
        ----------
        x : array-like, shape (n_modes,), complex or real

        Returns
        -------
        np.ndarray, shape (n_modes,), complex
        """
        x = np.asarray(x, dtype=complex)
        if x.shape != (self.n_modes,):
            raise ValueError(
                f"Expected input of shape ({self.n_modes},), got {x.shape}"
            )
        U = self._build_unitary()
        return U @ x

    @property
    def unitary(self) -> np.ndarray:
        """Current unitary matrix (n_modes x n_modes, complex)."""
        return self._build_unitary()
