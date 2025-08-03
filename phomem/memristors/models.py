"""
Physical models for memristive device behavior.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple, Optional
import chex
from functools import partial


class ConductanceModel:
    """Base conductance model for memristive devices."""
    
    def __init__(self, 
                 g_min: float = 1e-6,  # Minimum conductance (S)
                 g_max: float = 1e-3,  # Maximum conductance (S)
                 nonlinearity: float = 1.0):
        self.g_min = g_min
        self.g_max = g_max
        self.nonlinearity = nonlinearity
    
    def state_to_conductance(self, state: chex.Array) -> chex.Array:
        """Convert internal state variable to conductance."""
        # Sigmoid-like mapping with configurable nonlinearity
        normalized_state = jnp.clip(state, 0.0, 1.0)
        g_ratio = normalized_state**self.nonlinearity
        return self.g_min + (self.g_max - self.g_min) * g_ratio
    
    def conductance_to_state(self, conductance: chex.Array) -> chex.Array:
        """Convert conductance back to internal state."""
        g_ratio = (conductance - self.g_min) / (self.g_max - self.g_min)
        g_ratio = jnp.clip(g_ratio, 0.0, 1.0)
        return g_ratio**(1.0 / self.nonlinearity)


class SwitchingModel:
    """Voltage-driven switching dynamics for memristors."""
    
    def __init__(self,
                 v_set: float = 1.0,      # Set voltage (V)
                 v_reset: float = -1.5,   # Reset voltage (V)
                 switching_time: float = 1e-9,  # Characteristic time (s)
                 threshold_sharpness: float = 10.0):
        self.v_set = v_set
        self.v_reset = v_reset
        self.switching_time = switching_time
        self.threshold_sharpness = threshold_sharpness
    
    def switching_rate(self, voltage: chex.Array, state: chex.Array) -> chex.Array:
        """Calculate switching rate based on voltage and current state."""
        # Smooth threshold functions
        set_rate = (1.0 / self.switching_time) * jax.nn.sigmoid(
            self.threshold_sharpness * (voltage - self.v_set)
        ) * (1.0 - state)  # Only switch if not already set
        
        reset_rate = (1.0 / self.switching_time) * jax.nn.sigmoid(
            self.threshold_sharpness * (self.v_reset - voltage)
        ) * state  # Only switch if not already reset
        
        return set_rate - reset_rate
    
    def update_state(self, 
                     state: chex.Array, 
                     voltage: chex.Array, 
                     dt: float) -> chex.Array:
        """Update state based on applied voltage."""
        rate = self.switching_rate(voltage, state)
        new_state = state + rate * dt
        return jnp.clip(new_state, 0.0, 1.0)


class DriftModel:
    """Conductance drift model for long-term aging effects."""
    
    def __init__(self,
                 activation_energy: float = 0.6,  # eV
                 prefactor: float = 1e-6,
                 temperature_dependence: str = 'arrhenius'):
        self.activation_energy = activation_energy
        self.prefactor = prefactor
        self.temperature_dependence = temperature_dependence
        
        # Physical constants
        self.k_b = 8.617e-5  # Boltzmann constant (eV/K)
    
    def drift_coefficient(self, temperature: float) -> float:
        """Calculate temperature-dependent drift coefficient."""
        if self.temperature_dependence == 'arrhenius':
            return self.prefactor * jnp.exp(
                -self.activation_energy / (self.k_b * temperature)
            )
        else:
            return self.prefactor
    
    def apply_aging(self,
                    initial_conductance: chex.Array,
                    time_hours: float,
                    temperature: float = 300.0) -> chex.Array:
        """Apply drift aging over specified time."""
        drift_coeff = self.drift_coefficient(temperature)
        
        # Power-law drift: G(t) = G₀ * (1 + t/t₀)^(-α)
        alpha = 0.1  # Typical drift exponent
        t_0 = 1.0 / drift_coeff  # Characteristic time
        
        aging_factor = (1 + time_hours / t_0)**(-alpha)
        return initial_conductance * aging_factor


class AgingModel:
    """Comprehensive aging model including cycling and retention."""
    
    def __init__(self,
                 endurance_cycles: int = 1e6,
                 retention_time: float = 10 * 365 * 24,  # 10 years in hours
                 temperature_acceleration: float = 2.0):
        self.endurance_cycles = endurance_cycles
        self.retention_time = retention_time
        self.temperature_acceleration = temperature_acceleration
        
        # Aging mechanisms
        self.drift_model = DriftModel()
    
    def cycling_degradation(self, 
                           conductance: chex.Array,
                           cycle_count: int) -> chex.Array:
        """Model conductance degradation from write/erase cycling."""
        # Exponential degradation with cycling
        degradation_rate = 1.0 / self.endurance_cycles
        degradation_factor = jnp.exp(-degradation_rate * cycle_count)
        
        # Asymmetric degradation (more impact on high conductance)
        g_normalized = (conductance - self.drift_model.prefactor) / (1e-3 - self.drift_model.prefactor)
        asymmetry = 1.0 + 0.5 * g_normalized  # Higher states degrade faster
        
        return conductance * degradation_factor * asymmetry
    
    def retention_loss(self,
                       conductance: chex.Array,
                       time_hours: float,
                       temperature: float = 300.0) -> chex.Array:
        """Model conductance drift during retention."""
        # Use drift model with temperature acceleration
        effective_time = time_hours * (self.temperature_acceleration**(
            (temperature - 300) / 10
        ))
        
        return self.drift_model.apply_aging(
            conductance, effective_time, temperature
        )
    
    def comprehensive_aging(self,
                           initial_conductance: chex.Array,
                           cycle_count: int,
                           retention_hours: float,
                           temperature: float = 300.0) -> chex.Array:
        """Apply both cycling and retention aging."""
        # Apply cycling degradation first
        cycled_conductance = self.cycling_degradation(
            initial_conductance, cycle_count
        )
        
        # Then apply retention drift
        aged_conductance = self.retention_loss(
            cycled_conductance, retention_hours, temperature
        )
        
        return aged_conductance


class PCMModel(ConductanceModel):
    """Phase-Change Memory specific conductance model."""
    
    def __init__(self,
                 material: str = 'GST225',
                 geometry: str = 'mushroom',
                 dimensions: Dict[str, float] = None):
        # Material-specific parameters
        if material == 'GST225':
            g_min = 1e-8  # Amorphous state
            g_max = 1e-3  # Crystalline state
            nonlinearity = 2.0  # Stronger nonlinearity
        else:
            g_min = 1e-6
            g_max = 1e-3
            nonlinearity = 1.0
        
        super().__init__(g_min, g_max, nonlinearity)
        
        self.material = material
        self.geometry = geometry
        self.dimensions = dimensions or {
            'heater_radius': 50e-9,
            'thickness': 100e-9
        }
        
        # PCM-specific switching model
        self.switching_model = SwitchingModel(
            v_set=3.0,      # Higher voltage for crystallization
            v_reset=1.5,    # Lower for amorphization
            switching_time=100e-9,  # Slower than RRAM
            threshold_sharpness=5.0
        )
    
    def thermal_switching(self,
                         current: chex.Array,
                         pulse_width: float) -> chex.Array:
        """Model thermal switching dynamics."""
        # Joule heating: P = I²R
        resistance = 1.0 / self.state_to_conductance(0.5)  # Mid-state resistance
        power = current**2 * resistance
        
        # Temperature rise (simplified thermal model)
        thermal_resistance = 1e6  # K/W (very rough estimate)
        temperature_rise = power * thermal_resistance
        
        # Crystallization temperature threshold
        T_crystallization = 600  # K (above melting point)
        T_amorphization = 900    # K (quench from liquid)
        
        # State change probability based on temperature and time
        crystallize_prob = jax.nn.sigmoid(
            (temperature_rise - T_crystallization) / 50
        ) * pulse_width / 1e-9
        
        amorphize_prob = jax.nn.sigmoid(
            (temperature_rise - T_amorphization) / 50
        ) * pulse_width / 1e-9
        
        return crystallize_prob - amorphize_prob


class RRAMModel(ConductanceModel):
    """Resistive RAM specific conductance model."""
    
    def __init__(self,
                 oxide: str = 'HfO2',
                 thickness: float = 5e-9,
                 area: float = 100e-9**2):
        # RRAM typically has higher conductance range
        super().__init__(
            g_min=1e-5,
            g_max=1e-2,
            nonlinearity=1.5
        )
        
        self.oxide = oxide
        self.thickness = thickness
        self.area = area
        
        # RRAM-specific switching
        self.switching_model = SwitchingModel(
            v_set=1.2,
            v_reset=-1.0,
            switching_time=1e-9,  # Fast switching
            threshold_sharpness=20.0  # Sharp threshold
        )
        
        # Forming voltage (higher than set/reset)
        self.forming_voltage = 2.5
        self.is_formed = False
    
    def forming_process(self, voltage: chex.Array) -> bool:
        """Model initial forming of conductive filament."""
        if jnp.abs(voltage) > self.forming_voltage:
            self.is_formed = True
        return self.is_formed
    
    def filament_dynamics(self,
                         voltage: chex.Array,
                         state: chex.Array) -> chex.Array:
        """Model conductive filament growth/dissolution."""
        # Electric field in oxide
        field = voltage / self.thickness
        
        # Field-driven ion migration
        field_threshold = 1e8  # V/m
        migration_rate = jnp.tanh(field / field_threshold)
        
        # State represents filament connectivity (0=broken, 1=connected)
        if voltage > 0:  # Positive bias - filament growth
            growth_rate = migration_rate * (1 - state)
            return state + growth_rate * 1e-9  # Time step
        else:  # Negative bias - filament dissolution
            dissolution_rate = migration_rate * state
            return state - dissolution_rate * 1e-9