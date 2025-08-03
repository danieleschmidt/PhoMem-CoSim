"""
Core photonic components with differentiable physics models.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple
import chex
from functools import partial


@chex.dataclass
class PhotonicComponent:
    """Base class for all photonic components."""
    wavelength: float = 1550e-9  # Default telecom wavelength
    
    def __call__(self, inputs: chex.Array, params: Dict[str, Any]) -> chex.Array:
        raise NotImplementedError


class BeamSplitter(PhotonicComponent):
    """50:50 beam splitter with phase and loss."""
    
    def __init__(self, splitting_ratio: float = 0.5, loss_db: float = 0.1):
        self.splitting_ratio = splitting_ratio
        self.loss_db = loss_db
        self.loss_linear = 10**(-loss_db/10)
    
    def __call__(self, inputs: chex.Array, params: Dict[str, Any]) -> chex.Array:
        """
        Apply beam splitter transfer matrix.
        
        Args:
            inputs: Complex amplitudes [2,] for two input ports
            params: Parameters dict (unused for passive BS)
            
        Returns:
            Complex amplitudes [2,] for two output ports
        """
        chex.assert_shape(inputs, (2,))
        
        # 50:50 beam splitter matrix with loss
        t = jnp.sqrt(self.splitting_ratio * self.loss_linear)
        r = 1j * jnp.sqrt((1 - self.splitting_ratio) * self.loss_linear)
        
        bs_matrix = jnp.array([[t, r], [r, t]], dtype=jnp.complex64)
        
        return bs_matrix @ inputs


class PhaseShifter(PhotonicComponent):
    """Generic phase shifter with configurable physics model."""
    
    def __init__(self, phase_type: str = "thermal", loss_db: float = 0.05):
        self.phase_type = phase_type
        self.loss_db = loss_db
        self.loss_linear = 10**(-loss_db/10)
    
    def __call__(self, inputs: chex.Array, params: Dict[str, Any]) -> chex.Array:
        """
        Apply phase shift with loss.
        
        Args:
            inputs: Complex amplitude (scalar or array)
            params: Must contain 'phase' key
            
        Returns:
            Phase-shifted complex amplitude
        """
        phase = params.get('phase', 0.0)
        
        # Apply phase shift and insertion loss
        phase_factor = jnp.exp(1j * phase) * jnp.sqrt(self.loss_linear)
        
        return inputs * phase_factor


class MachZehnderInterferometer(PhotonicComponent):
    """Mach-Zehnder interferometer with differential phase control."""
    
    def __init__(self, 
                 coupling_ratio: float = 0.5,
                 loss_db: float = 0.2,
                 phase_type: str = "thermal"):
        self.coupling_ratio = coupling_ratio
        self.loss_db = loss_db
        self.loss_linear = 10**(-loss_db/10)
        self.phase_type = phase_type
        
        # Initialize beam splitters
        self.input_bs = BeamSplitter(coupling_ratio, loss_db/2)
        self.output_bs = BeamSplitter(coupling_ratio, loss_db/2)
        self.phase_shifter = PhaseShifter(phase_type, loss_db/4)
    
    def __call__(self, inputs: chex.Array, params: Dict[str, Any]) -> chex.Array:
        """
        MZI transfer function: T = cos²(φ/2), R = sin²(φ/2)
        
        Args:
            inputs: Complex amplitudes [2,] for two input ports
            params: Must contain 'phase' for differential phase
            
        Returns:
            Complex amplitudes [2,] for two output ports
        """
        chex.assert_shape(inputs, (2,))
        
        # First beam splitter
        mid_fields = self.input_bs(inputs, {})
        
        # Apply differential phase shift (only on upper arm)
        phase_params = {'phase': params.get('phase', 0.0)}
        mid_fields = mid_fields.at[0].set(
            self.phase_shifter(mid_fields[0], phase_params)
        )
        
        # Second beam splitter  
        outputs = self.output_bs(mid_fields, {})
        
        return outputs
    
    def get_transmission(self, phase: float) -> Tuple[float, float]:
        """Get transmission and reflection coefficients."""
        T = jnp.cos(phase/2)**2 * self.loss_linear
        R = jnp.sin(phase/2)**2 * self.loss_linear
        return T, R