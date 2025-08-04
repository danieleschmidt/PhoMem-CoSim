"""
High-level photonic devices and arrays.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple, Optional
import chex
from functools import partial

from .components import MachZehnderInterferometer, PhaseShifter
from ..utils.exceptions import (
    PhotonicError, PhaseShifterError, InputValidationError,
    validate_array_input, validate_range, handle_jax_errors
)
from ..utils.logging import get_logger


class MachZehnderMesh:
    """Triangular mesh of Mach-Zehnder interferometers for unitary operations."""
    
    def __init__(self, 
                 size: int,
                 wavelength: float = 1550e-9,
                 loss_db_cm: float = 0.5,
                 phase_shifter: str = 'thermal'):
        
        # Input validation
        validate_range(size, 'size', min_val=2, max_val=128)
        validate_range(wavelength, 'wavelength', min_val=1.0e-6, max_val=2.0e-6)
        validate_range(loss_db_cm, 'loss_db_cm', min_val=0.0, max_val=100.0)
        
        valid_phase_shifters = ['thermal', 'plasma', 'pcm']
        if phase_shifter not in valid_phase_shifters:
            raise InputValidationError(
                f"Invalid phase shifter type '{phase_shifter}'",
                parameter_name='phase_shifter',
                parameter_value=phase_shifter,
                suggestions=[f"Use one of: {valid_phase_shifters}"]
            )
        
        self.size = size
        self.wavelength = wavelength  
        self.loss_db_cm = loss_db_cm
        self.phase_shifter_type = phase_shifter
        self.logger = get_logger('photonics.mzi_mesh')
        
        # Calculate number of MZIs needed for triangular mesh
        self.n_mzis = size * (size - 1) // 2
        
        self.logger.info(f"Initializing {size}x{size} MZI mesh with {self.n_mzis} interferometers")
        
        try:
            # Initialize MZI components
            self.mzis = [MachZehnderInterferometer(
                coupling_ratio=0.5,
                loss_db=loss_db_cm * 0.1,  # Assume 1mm path length
                phase_type=phase_shifter
            ) for _ in range(self.n_mzis)]
        except Exception as e:
            raise PhotonicError(
                f"Failed to initialize MZI components: {str(e)}",
                context={'size': size, 'n_mzis': self.n_mzis}
            )
    
    @handle_jax_errors
    def __call__(self, inputs: chex.Array, params: Dict[str, Any]) -> chex.Array:
        """
        Apply triangular MZI mesh transformation.
        
        Args:
            inputs: Complex input amplitudes [size,]
            params: Dict with 'phases' key containing [n_mzis,] phase values
            
        Returns:
            Complex output amplitudes [size,]
        """
        try:
            # Input validation
            validate_array_input(inputs, 'inputs', expected_shape=(self.size,))
            
            if not isinstance(params, dict):
                raise InputValidationError(
                    "params must be a dictionary",
                    parameter_name='params',
                    parameter_value=type(params).__name__,
                    expected_type=dict
                )
            
            phases = params.get('phases', jnp.zeros(self.n_mzis))
            validate_array_input(phases, 'phases', expected_shape=(self.n_mzis,))
            
            # Validate phase ranges
            if jnp.any(jnp.abs(phases) > 2 * jnp.pi):
                self.logger.warning(f"Large phase values detected: max={jnp.max(jnp.abs(phases)):.2f} rad")
                
            # Check for NaN or infinite values
            if jnp.any(jnp.isnan(inputs)) or jnp.any(jnp.isinf(inputs)):
                raise InputValidationError(
                    "Input contains NaN or infinite values",
                    parameter_name='inputs',
                    parameter_value="contains NaN/inf"
                )
            
            if jnp.any(jnp.isnan(phases)) or jnp.any(jnp.isinf(phases)):
                raise PhaseShifterError(
                    "Phase parameters contain NaN or infinite values",
                    context={'phases_stats': {'min': float(jnp.min(phases)), 'max': float(jnp.max(phases))}}
                )
            
            # Initialize field vector
            fields = inputs.astype(jnp.complex64)
            
            # Apply MZIs in triangular pattern
            mzi_idx = 0
            for layer in range(self.size - 1):
                for pos in range(self.size - 1 - layer):
                    try:
                        # Get the two modes to interfere
                        mode_pair = jnp.array([fields[pos], fields[pos + 1]])
                        
                        # Apply MZI with current phase
                        phase_params = {'phase': phases[mzi_idx]}
                        output_pair = self.mzis[mzi_idx](mode_pair, phase_params)
                        
                        # Update field vector
                        fields = fields.at[pos].set(output_pair[0])
                        fields = fields.at[pos + 1].set(output_pair[1])
                        
                        mzi_idx += 1
                        
                    except Exception as e:
                        raise PhotonicError(
                            f"Error in MZI {mzi_idx} at layer {layer}, position {pos}: {str(e)}",
                            context={
                                'mzi_index': mzi_idx,
                                'layer': layer,
                                'position': pos,
                                'phase_value': float(phases[mzi_idx]) if mzi_idx < len(phases) else None
                            }
                        )
            
            # Validate output
            if jnp.any(jnp.isnan(fields)) or jnp.any(jnp.isinf(fields)):
                raise PhotonicError(
                    "Output contains NaN or infinite values",
                    context={
                        'input_power': float(jnp.sum(jnp.abs(inputs)**2)),
                        'output_power': float(jnp.sum(jnp.abs(fields)**2)),
                        'phase_range': [float(jnp.min(phases)), float(jnp.max(phases))]
                    },
                    suggestions=[
                        "Check input power levels",
                        "Verify phase shifter settings",
                        "Check for component failures"
                    ]
                )
            
            return fields
            
        except Exception as e:
            if isinstance(e, (PhotonicError, InputValidationError, PhaseShifterError)):
                raise
            else:
                raise PhotonicError(
                    f"Unexpected error in MZI mesh: {str(e)}",
                    context={'mesh_size': self.size, 'n_mzis': self.n_mzis}
                )
    
    def get_unitary_matrix(self, phases: chex.Array) -> chex.Array:
        """Get the unitary matrix representation."""
        # Use canonical basis to build matrix
        identity = jnp.eye(self.size, dtype=jnp.complex64)
        matrix_cols = []
        
        for i in range(self.size):
            basis_vector = identity[:, i]
            output = self(basis_vector, {'phases': phases})
            matrix_cols.append(output)
            
        return jnp.stack(matrix_cols, axis=1)


class ThermalPhaseShifter:
    """Thermal phase shifter with realistic power consumption."""
    
    def __init__(self, 
                 power_per_pi: float = 20e-3,  # 20mW for π shift
                 response_time: float = 10e-6,  # 10μs
                 efficiency: float = 0.8):
        self.power_per_pi = power_per_pi
        self.response_time = response_time
        self.efficiency = efficiency
        
        # Thermal constants
        self.thermo_optic_coeff = 1.86e-4  # /K for silicon
        self.thermal_resistance = 1000  # K/W
    
    def phase_to_power(self, phase: float) -> float:
        """Convert phase to required electrical power."""
        return jnp.abs(phase) / jnp.pi * self.power_per_pi / self.efficiency
    
    def get_temperature_rise(self, phase: float) -> float:
        """Calculate temperature rise for given phase."""
        power = self.phase_to_power(phase)
        return power * self.thermal_resistance
    
    def __call__(self, inputs: chex.Array, params: Dict[str, Any]) -> chex.Array:
        """Apply thermal phase shift."""
        phase = params.get('phase', 0.0)
        loss_factor = params.get('loss_factor', 1.0)
        
        return inputs * jnp.exp(1j * phase) * jnp.sqrt(loss_factor)


class PlasmaDispersionPhaseShifter:
    """Plasma dispersion phase shifter for high-speed operation."""
    
    def __init__(self,
                 voltage_per_pi: float = 5.0,  # 5V for π shift
                 capacitance: float = 10e-15,  # 10fF
                 bandwidth: float = 50e9):  # 50GHz
        self.voltage_per_pi = voltage_per_pi
        self.capacitance = capacitance
        self.bandwidth = bandwidth
    
    def phase_to_power(self, phase: float, frequency: float) -> float:
        """Calculate dynamic power consumption."""
        voltage = jnp.abs(phase) / jnp.pi * self.voltage_per_pi
        return self.capacitance * voltage**2 * frequency
    
    def __call__(self, inputs: chex.Array, params: Dict[str, Any]) -> chex.Array:
        """Apply plasma dispersion phase shift."""
        phase = params.get('phase', 0.0)
        loss_db = params.get('loss_db', 0.1)
        loss_factor = 10**(-loss_db/10)
        
        return inputs * jnp.exp(1j * phase) * jnp.sqrt(loss_factor)


class PCMPhaseShifter:
    """Phase-change material phase shifter with non-volatile operation."""
    
    def __init__(self,
                 material: str = 'GST225',
                 switching_energy: float = 100e-12):  # 100pJ
        self.material = material
        self.switching_energy = switching_energy
        
        # Material properties for GST225
        if material == 'GST225':
            self.n_amorphous = 4.0 + 0.05j
            self.n_crystalline = 6.5 + 0.3j
            self.thickness = 50e-9  # 50nm
    
    def get_phase_shift(self, crystalline_fraction: float) -> float:
        """Calculate phase shift based on crystalline fraction."""
        n_eff = (self.n_amorphous * (1 - crystalline_fraction) + 
                 self.n_crystalline * crystalline_fraction)
        
        phase_shift = 2 * jnp.pi * n_eff.real * self.thickness / 1550e-9
        return phase_shift
    
    def __call__(self, inputs: chex.Array, params: Dict[str, Any]) -> chex.Array:
        """Apply PCM phase shift."""
        crystalline_fraction = params.get('crystalline_fraction', 0.0)
        phase = self.get_phase_shift(crystalline_fraction)
        
        # Calculate absorption loss
        n_eff = (self.n_amorphous * (1 - crystalline_fraction) + 
                 self.n_crystalline * crystalline_fraction)
        absorption = jnp.exp(-2 * jnp.pi * n_eff.imag * self.thickness / 1550e-9)
        
        return inputs * jnp.exp(1j * phase) * absorption


class PhotoDetectorArray:
    """Array of photodetectors with realistic noise models."""
    
    def __init__(self,
                 responsivity: float = 0.8,  # A/W
                 dark_current: float = 1e-9,  # 1nA
                 thermal_noise: bool = True,
                 shot_noise: bool = True):
        self.responsivity = responsivity
        self.dark_current = dark_current
        self.thermal_noise = thermal_noise
        self.shot_noise = shot_noise
        
        # Noise parameters
        self.kT = 4.14e-21  # kT at 300K in J
        self.load_resistance = 50  # Ohm
        self.bandwidth = 10e9  # 10GHz
    
    def __call__(self, 
                 optical_inputs: chex.Array, 
                 params: Dict[str, Any],
                 key: Optional[chex.PRNGKey] = None) -> chex.Array:
        """
        Convert optical power to electrical current with noise.
        
        Args:
            optical_inputs: Complex optical amplitudes
            params: Additional parameters
            key: PRNG key for noise generation
            
        Returns:
            Electrical currents [A]
        """
        # Calculate optical power
        optical_power = jnp.abs(optical_inputs)**2
        
        # Convert to electrical current
        photocurrent = optical_power * self.responsivity
        
        # Add dark current
        total_current = photocurrent + self.dark_current
        
        # Add noise if requested and key provided
        if key is not None:
            if self.shot_noise:
                # Shot noise: σ² = 2qI·BW (Poisson)
                shot_variance = 2 * 1.602e-19 * total_current * self.bandwidth
                key, subkey = jax.random.split(key)
                shot_noise = jax.random.normal(subkey, total_current.shape) * jnp.sqrt(shot_variance)
                total_current += shot_noise
            
            if self.thermal_noise:
                # Thermal noise: σ² = 4kT·BW/R (Johnson)
                thermal_variance = 4 * self.kT * self.bandwidth / self.load_resistance
                key, subkey = jax.random.split(key)
                thermal_noise = jax.random.normal(subkey, total_current.shape) * jnp.sqrt(thermal_variance)
                total_current += thermal_noise
        
        return total_current


class SiliconWaveguide:
    """Silicon photonic waveguide with realistic propagation loss."""
    
    def __init__(self,
                 width: float = 450e-9,
                 height: float = 220e-9,
                 roughness_nm: float = 3.0):
        self.width = width
        self.height = height
        self.roughness_nm = roughness_nm
        
        # Calculate effective index (simplified)
        self.n_eff = 2.4  # Typical for 450x220nm Si waveguide
        
        # Scattering loss from sidewall roughness
        self.scattering_loss_db_cm = self._calculate_scattering_loss()
    
    def _calculate_scattering_loss(self) -> float:
        """Calculate scattering loss from sidewall roughness."""
        # Simplified Payne-Lacey model
        sigma = self.roughness_nm * 1e-9
        correlation_length = 50e-9  # Typical
        
        # Loss coefficient (dB/cm)
        loss_coeff = 10 * jnp.log10(jnp.e) * (
            (2 * jnp.pi * self.n_eff * sigma / 1550e-9)**2 *
            correlation_length / self.width
        )
        
        return jnp.clip(loss_coeff, 0.1, 10.0)  # Reasonable bounds
    
    def propagate(self, 
                  inputs: chex.Array, 
                  length: float,
                  params: Dict[str, Any]) -> chex.Array:
        """Propagate light through waveguide segment."""
        # Phase accumulation
        phase = 2 * jnp.pi * self.n_eff * length / 1550e-9
        
        # Loss (dB to linear)
        loss_linear = 10**(-self.scattering_loss_db_cm * length * 100 / 10)
        
        return inputs * jnp.exp(1j * phase) * jnp.sqrt(loss_linear)


class SiliconNitrideWaveguide:
    """Silicon nitride waveguide with ultra-low loss."""
    
    def __init__(self,
                 width: float = 800e-9,
                 height: float = 400e-9):
        self.width = width
        self.height = height
        
        # SiN properties
        self.n_eff = 1.9  # Lower than silicon
        self.loss_db_cm = 0.01  # Ultra-low loss
    
    def propagate(self, 
                  inputs: chex.Array, 
                  length: float,
                  params: Dict[str, Any]) -> chex.Array:
        """Propagate light through SiN waveguide."""
        phase = 2 * jnp.pi * self.n_eff * length / 1550e-9
        loss_linear = 10**(-self.loss_db_cm * length * 100 / 10)
        
        return inputs * jnp.exp(1j * phase) * jnp.sqrt(loss_linear)