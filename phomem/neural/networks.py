"""
Core hybrid network architectures.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
import chex
from functools import partial
import flax.linen as nn

from ..photonics import MachZehnderMesh, PhotoDetectorArray
from ..memristors import PCMCrossbar, RRAMCrossbar


class PhotonicLayer(nn.Module):
    """Photonic neural network layer using MZI mesh."""
    
    size: int
    wavelength: float = 1550e-9
    loss_db_cm: float = 0.5
    phase_shifter_type: str = 'thermal'
    include_nonlinearity: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        # Input validation
        if self.size <= 0:
            raise ValueError(f"Size must be positive, got {self.size}")
        if self.wavelength <= 0:
            raise ValueError(f"Wavelength must be positive, got {self.wavelength}")
        if self.loss_db_cm < 0:
            raise ValueError(f"Loss must be non-negative, got {self.loss_db_cm}")
        if self.phase_shifter_type not in ['thermal', 'plasma', 'pcm']:
            raise ValueError(f"Invalid phase shifter type: {self.phase_shifter_type}")
    
    def setup(self):
        self.mzi_mesh = MachZehnderMesh(
            size=self.size,
            wavelength=self.wavelength,
            loss_db_cm=self.loss_db_cm,
            phase_shifter=self.phase_shifter_type
        )
        
        if self.include_nonlinearity:
            # Kerr nonlinearity parameters
            self.nonlinear_coeff = 2.6e-20  # m²/W for silicon
    
    @nn.compact
    def __call__(self, inputs: chex.Array, training: bool = True) -> chex.Array:
        """
        Forward pass through photonic layer.
        
        Args:
            inputs: Complex optical amplitudes [batch, size] or [size]
            training: Whether in training mode
            
        Returns:
            Complex optical outputs [batch, size] or [size]
        """
        # Handle batch dimension
        batch_size = None
        if inputs.ndim == 2:
            batch_size, _ = inputs.shape
            chex.assert_shape(inputs, (batch_size, self.size))
        else:
            chex.assert_shape(inputs, (self.size,))
        
        # Initialize phase parameters
        n_phases = self.size * (self.size - 1) // 2
        phases = self.param('phases', 
                           nn.initializers.uniform(scale=2*jnp.pi),
                           (n_phases,))
        
        # Apply MZI mesh transformation
        if batch_size is not None:
            # Batch processing
            outputs = jax.vmap(
                lambda x: self.mzi_mesh(x, {'phases': phases})
            )(inputs)
        else:
            outputs = self.mzi_mesh(inputs, {'phases': phases})
        
        # Apply optical nonlinearity if requested
        if self.include_nonlinearity:
            optical_power = jnp.abs(outputs)**2
            nonlinear_phase = self.nonlinear_coeff * optical_power * 1000  # Assume 1mm length
            outputs = outputs * jnp.exp(1j * nonlinear_phase)
        
        return outputs
    
    def get_optical_losses(self, params: Dict[str, Any]) -> float:
        """Calculate total optical losses in dB."""
        # Simplified loss model based on number of components
        n_mzis = self.size * (self.size - 1) // 2
        insertion_loss_per_mzi = self.loss_db_cm * 0.1  # Assume 1mm per MZI
        total_loss = n_mzis * insertion_loss_per_mzi
        return total_loss
    
    def get_power_dissipation(self, params: Dict[str, Any]) -> float:
        """Calculate power dissipation from phase shifters."""
        phases = params.get('phases', jnp.zeros(self.size * (self.size - 1) // 2))
        
        if self.phase_shifter_type == 'thermal':
            # 20mW per π phase shift
            power_per_pi = 20e-3
            total_power = jnp.sum(jnp.abs(phases)) / jnp.pi * power_per_pi
        elif self.phase_shifter_type == 'plasma':
            # Much lower static power, but depends on switching frequency
            total_power = len(phases) * 1e-6  # 1μW per phase shifter
        else:
            total_power = 0.0
        
        return total_power


class MemristiveLayer(nn.Module):
    """Memristive crossbar layer for synaptic weights."""
    
    input_size: int
    output_size: int
    device_type: str = 'PCM'  # 'PCM' or 'RRAM'
    include_aging: bool = False
    variability: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        # Input validation
        if self.input_size <= 0:
            raise ValueError(f"Input size must be positive, got {self.input_size}")
        if self.output_size <= 0:
            raise ValueError(f"Output size must be positive, got {self.output_size}")
        if self.device_type not in ['PCM', 'RRAM']:
            raise ValueError(f"Invalid device type: {self.device_type}")
    
    def setup(self):
        if self.device_type == 'PCM':
            self.crossbar = PCMCrossbar(
                rows=self.input_size,
                cols=self.output_size,
                variability=self.variability
            )
        else:
            self.crossbar = RRAMCrossbar(
                rows=self.input_size,
                cols=self.output_size
            )
    
    @nn.compact
    def __call__(self, inputs: chex.Array, training: bool = True) -> chex.Array:
        """
        Forward pass through memristive layer.
        
        Args:
            inputs: Input currents [batch, input_size] or [input_size]
            training: Whether in training mode
            
        Returns:
            Output currents [batch, output_size] or [output_size]
        """
        # Handle batch dimension
        batch_size = None
        if inputs.ndim == 2:
            batch_size, _ = inputs.shape
        
        # Initialize device states (conductance values)
        states = self.param('states',
                           lambda key, shape: jax.random.uniform(key, shape, minval=0.2, maxval=0.8),
                           (self.input_size, self.output_size))
        
        # Voltage parameters for crossbar operation
        voltage_compliance = self.param('voltage_compliance',
                                      lambda key, shape: jnp.full(shape, 0.5),
                                      ())
        
        # Convert inputs to crossbar voltages
        # Simplified: assume current-to-voltage conversion
        input_voltages = inputs * 1000  # Assume 1kΩ load resistance
        col_voltages = jnp.zeros(self.output_size)  # Ground columns
        
        crossbar_params = {
            'states': states,
            'include_aging': self.include_aging,
            'cycle_count': 0,
            'age_hours': 0.0
        }
        
        if batch_size is not None:
            # Batch processing
            outputs = jax.vmap(
                lambda row_v: self.crossbar(row_v, col_voltages, crossbar_params)
            )(input_voltages)
        else:
            outputs = self.crossbar(input_voltages, col_voltages, crossbar_params)
        
        return outputs
    
    def update_weights(self, 
                      target_states: chex.Array,
                      programming_scheme: str = 'iterative') -> chex.Array:
        """Update memristor states to target values."""
        return self.crossbar.write_array(target_states, programming_scheme)


class TransimpedanceAmplifier(nn.Module):
    """Transimpedance amplifier for current-to-voltage conversion."""
    
    gain: float = 1e5  # V/A
    bandwidth: float = 10e9  # Hz
    noise_current: float = 1e-12  # A/√Hz
    
    @nn.compact
    def __call__(self, 
                 currents: chex.Array,
                 training: bool = True,
                 key: Optional[chex.PRNGKey] = None) -> chex.Array:
        """
        Convert currents to voltages with amplification.
        
        Args:
            currents: Input currents [batch, size] or [size]
            training: Whether in training mode
            key: PRNG key for noise generation
            
        Returns:
            Output voltages [batch, size] or [size]
        """
        # Convert complex currents to real (magnitude)
        real_currents = jnp.real(currents) if jnp.iscomplexobj(currents) else currents
        
        # Simple gain multiplication
        voltages = real_currents * self.gain
        
        # Add noise if key provided
        if key is not None and training:
            noise_std = self.noise_current * jnp.sqrt(self.bandwidth) * self.gain
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, voltages.shape) * noise_std
            voltages = voltages + noise
        
        # Saturation limits (±5V typical)
        voltages = jnp.clip(voltages, -5.0, 5.0)
        
        return voltages


class HybridNetwork(nn.Module):
    """Complete hybrid photonic-memristive neural network."""
    
    layers: List[nn.Module]
    include_noise: bool = True
    
    def setup(self):
        self.network_layers = self.layers
    
    @nn.compact
    def __call__(self, 
                 inputs: chex.Array,
                 training: bool = True,
                 key: Optional[chex.PRNGKey] = None) -> chex.Array:
        """
        Forward pass through hybrid network.
        
        Args:
            inputs: Network inputs (optical or electrical)
            training: Whether in training mode
            key: PRNG key for noise generation
            
        Returns:
            Network outputs
        """
        x = inputs
        
        for i, layer in enumerate(self.network_layers):
            if key is not None:
                key, subkey = jax.random.split(key)
            else:
                subkey = None
            
            # Apply layer with appropriate parameters
            if isinstance(layer, (PhotonicLayer, MemristiveLayer)):
                x = layer(x, training=training)
            elif isinstance(layer, TransimpedanceAmplifier):
                x = layer(x, training=training, key=subkey)
            elif hasattr(layer, '__call__'):
                # Handle other layers (photodetectors, etc.)
                if 'key' in layer.__call__.__code__.co_varnames:
                    x = layer(x, key=subkey)
                else:
                    x = layer(x)
            else:
                # Fallback for simple functions
                x = layer(x)
        
        return x
    
    def get_optical_losses(self, params: Dict[str, Any]) -> float:
        """Get total optical losses across all photonic layers."""
        total_loss = 0.0
        for layer in self.network_layers:
            if isinstance(layer, PhotonicLayer):
                total_loss += layer.get_optical_losses(params)
        return total_loss
    
    def get_power_dissipation(self, params: Dict[str, Any]) -> float:
        """Get total power dissipation across all layers."""
        total_power = 0.0
        for layer in self.network_layers:
            if hasattr(layer, 'get_power_dissipation'):
                total_power += layer.get_power_dissipation(params)
        return total_power
    
    def estimate_lifetime_degradation(self, params: Dict[str, Any]) -> float:
        """Estimate lifetime degradation penalty."""
        # Simplified model based on power dissipation and optical losses
        power = self.get_power_dissipation(params)
        optical_loss = self.get_optical_losses(params)
        
        # Higher power and loss lead to faster aging
        aging_factor = (power / 1e-3)**0.5 + (optical_loss / 1.0)**0.5
        
        return aging_factor


def create_example_hybrid_network(input_size: int = 4, 
                                 hidden_size: int = 16,
                                 output_size: int = 10) -> HybridNetwork:
    """Create an example hybrid network architecture."""
    
    # Photonic front-end for matrix multiplication
    photonic_layer = PhotonicLayer(
        size=input_size,
        wavelength=1550e-9,
        phase_shifter_type='thermal'
    )
    
    # Photodetector array for O/E conversion
    photodetector = PhotoDetectorArray(
        responsivity=0.8,
        dark_current=1e-9,
        thermal_noise=True,
        shot_noise=True
    )
    
    # Memristive layer for nonlinear processing
    memristive_layer = MemristiveLayer(
        input_size=input_size,
        output_size=hidden_size,
        device_type='PCM',
        include_aging=False
    )
    
    # Transimpedance amplifier
    tia = TransimpedanceAmplifier(gain=1e5)
    
    # Output memristive layer
    output_layer = MemristiveLayer(
        input_size=hidden_size,
        output_size=output_size,
        device_type='RRAM'
    )
    
    # Combine layers
    layers = [
        photonic_layer,
        # Note: photodetector needs special handling for optical->electrical conversion
        memristive_layer,
        tia,
        output_layer
    ]
    
    return HybridNetwork(layers=layers)