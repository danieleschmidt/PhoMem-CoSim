"""
High-level memristive devices and crossbar arrays.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import chex
from functools import partial

from .models import PCMModel, RRAMModel, DriftModel, AgingModel


class PCMDevice:
    """Individual Phase-Change Memory device."""
    
    def __init__(self,
                 material: str = 'GST225',
                 geometry: str = 'mushroom',
                 dimensions: Dict[str, float] = None,
                 temperature: float = 300.0):
        self.material = material
        self.geometry = geometry
        self.temperature = temperature
        
        # Initialize physics models
        self.pcm_model = PCMModel(material, geometry, dimensions)
        self.aging_model = AgingModel()
        
        # Device state
        self.state = 0.0  # 0=amorphous, 1=crystalline
        self.cycle_count = 0
        self.age_hours = 0.0
    
    def __call__(self, 
                 voltage: chex.Array,
                 params: Dict[str, Any]) -> chex.Array:
        """
        Compute current response to applied voltage.
        
        Args:
            voltage: Applied voltage (V)
            params: Device parameters including 'state'
            
        Returns:
            Current (A) based on Ohm's law
        """
        state = params.get('state', self.state)
        
        # Get conductance from state
        conductance = self.pcm_model.state_to_conductance(state)
        
        # Apply aging effects if requested
        if params.get('include_aging', False):
            conductance = self.aging_model.comprehensive_aging(
                conductance,
                params.get('cycle_count', self.cycle_count),
                params.get('age_hours', self.age_hours),
                self.temperature
            )
        
        # Ohm's law with possible nonlinearity
        nonlinearity = params.get('nonlinearity', 1.0)
        current = conductance * voltage * jnp.sign(voltage)**(nonlinearity - 1)
        
        return current
    
    def write_pulse(self, 
                    voltage: float, 
                    pulse_width: float,
                    pulse_type: str = 'voltage') -> float:
        """Apply write pulse and return new state."""
        if pulse_type == 'voltage':
            # Voltage-controlled switching
            new_state = self.pcm_model.switching_model.update_state(
                self.state, voltage, pulse_width
            )
        elif pulse_type == 'current':
            # Current-controlled (thermal) switching
            delta_state = self.pcm_model.thermal_switching(voltage, pulse_width)
            new_state = jnp.clip(self.state + delta_state, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown pulse type: {pulse_type}")
        
        self.state = new_state
        self.cycle_count += 1
        return new_state
    
    def read_conductance(self, read_voltage: float = 0.1) -> float:
        """Read current conductance with small read voltage."""
        conductance = self.pcm_model.state_to_conductance(self.state)
        
        # Apply aging effects
        aged_conductance = self.aging_model.comprehensive_aging(
            conductance, self.cycle_count, self.age_hours, self.temperature
        )
        
        return aged_conductance


class RRAMDevice:
    """Individual Resistive RAM device."""
    
    def __init__(self,
                 oxide: str = 'HfO2',
                 thickness: float = 5e-9,
                 area: float = 100e-9**2,
                 forming_voltage: float = 2.5,
                 temperature: float = 300.0):
        self.oxide = oxide
        self.thickness = thickness
        self.area = area
        self.forming_voltage = forming_voltage
        self.temperature = temperature
        
        # Initialize physics models
        self.rram_model = RRAMModel(oxide, thickness, area)
        self.aging_model = AgingModel(
            endurance_cycles=1e8,  # Higher endurance than PCM
            retention_time=10 * 365 * 24
        )
        
        # Device state
        self.state = 0.0  # 0=HRS, 1=LRS
        self.is_formed = False
        self.cycle_count = 0
        self.age_hours = 0.0
    
    def __call__(self, 
                 voltage: chex.Array,
                 params: Dict[str, Any]) -> chex.Array:
        """Compute current response including forming."""
        state = params.get('state', self.state)
        is_formed = params.get('is_formed', self.is_formed)
        
        # Check if forming is needed
        if not is_formed and jnp.abs(voltage) > self.forming_voltage:
            is_formed = True
            state = 0.5  # Partial formation
        
        if not is_formed:
            # Virgin device - very high resistance
            return voltage * 1e-12  # pA current
        
        # Formed device - normal operation
        conductance = self.rram_model.state_to_conductance(state)
        
        # Apply aging if requested
        if params.get('include_aging', False):
            conductance = self.aging_model.comprehensive_aging(
                conductance,
                params.get('cycle_count', self.cycle_count),
                params.get('age_hours', self.age_hours),
                self.temperature
            )
        
        # I-V characteristic with possible nonlinearity
        nonlinearity = params.get('nonlinearity', 1.2)
        current = conductance * voltage * jnp.abs(voltage)**(nonlinearity - 1)
        
        return current
    
    def write_pulse(self, voltage: float, pulse_width: float) -> float:
        """Apply write pulse and update state."""
        # Check forming first
        if not self.is_formed:
            self.is_formed = self.rram_model.forming_process(voltage)
            if self.is_formed:
                self.state = 0.5  # Initial formed state
        
        if self.is_formed:
            # Normal switching
            self.state = self.rram_model.switching_model.update_state(
                self.state, voltage, pulse_width
            )
            self.cycle_count += 1
        
        return self.state


class PCMCrossbar:
    """PCM crossbar array with realistic device variations."""
    
    def __init__(self,
                 rows: int,
                 cols: int,
                 device_model: str = 'pcm_mushroom',
                 temperature: float = 300.0,
                 variability: bool = True):
        self.rows = rows
        self.cols = cols
        self.device_model = device_model
        self.temperature = temperature
        self.variability = variability
        
        # Initialize device array
        self.devices = []
        for i in range(rows):
            row_devices = []
            for j in range(cols):
                device = PCMDevice(
                    material='GST225',
                    geometry='mushroom',
                    temperature=temperature
                )
                row_devices.append(device)
            self.devices.append(row_devices)
        
        # Device variability parameters
        if variability:
            self.device_variations = self._generate_variations()
        else:
            self.device_variations = jnp.ones((rows, cols))
    
    def _generate_variations(self, key: Optional[chex.PRNGKey] = None) -> chex.Array:
        """Generate device-to-device variations."""
        if key is None:
            key = jax.random.PRNGKey(42)
        
        # Log-normal distribution for conductance variations (typical 15% CV)
        cv = 0.15  # Coefficient of variation
        sigma = jnp.sqrt(jnp.log(1 + cv**2))
        mu = -sigma**2 / 2  # Ensure mean = 1
        
        variations = jax.random.lognormal(key, sigma, (self.rows, self.cols))
        return variations / jnp.mean(variations)  # Normalize to mean=1
    
    def __call__(self, 
                 row_voltages: chex.Array,
                 col_voltages: chex.Array,
                 params: Dict[str, Any]) -> chex.Array:
        """
        Compute crossbar currents using Kirchhoff's laws.
        
        Args:
            row_voltages: Voltages applied to rows [rows,]
            col_voltages: Voltages applied to columns [cols,]
            params: Parameters including device states
            
        Returns:
            Column currents [cols,]
        """
        chex.assert_shape(row_voltages, (self.rows,))
        chex.assert_shape(col_voltages, (self.cols,))
        
        # Get device states (default to mid-range)
        states = params.get('states', jnp.ones((self.rows, self.cols)) * 0.5)
        chex.assert_shape(states, (self.rows, self.cols))
        
        # Calculate voltage across each device
        voltage_matrix = row_voltages[:, None] - col_voltages[None, :]
        
        # Get conductance matrix
        conductance_matrix = jnp.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                device_params = {
                    'state': states[i, j],
                    'include_aging': params.get('include_aging', False),
                    'cycle_count': params.get('cycle_count', 0),
                    'age_hours': params.get('age_hours', 0.0)
                }
                
                # Get base conductance
                base_conductance = self.devices[i][j].pcm_model.state_to_conductance(
                    states[i, j]
                )
                
                # Apply device variations
                varied_conductance = base_conductance * self.device_variations[i, j]
                
                conductance_matrix = conductance_matrix.at[i, j].set(varied_conductance)
        
        # Solve for currents using conductance matrix
        # Simplified: assume ideal voltage sources (no wire resistance)
        device_currents = conductance_matrix * voltage_matrix
        
        # Sum currents flowing into each column
        col_currents = jnp.sum(device_currents, axis=0)
        
        return col_currents
    
    def write_array(self, 
                    target_states: chex.Array,
                    programming_scheme: str = 'iterative') -> chex.Array:
        """Program the crossbar to target conductance states."""
        chex.assert_shape(target_states, (self.rows, self.cols))
        
        if programming_scheme == 'iterative':
            return self._iterative_programming(target_states)
        elif programming_scheme == 'direct':
            return self._direct_programming(target_states)
        else:
            raise ValueError(f"Unknown programming scheme: {programming_scheme}")
    
    def _iterative_programming(self, target_states: chex.Array) -> chex.Array:
        """Iterative write-and-verify programming."""
        current_states = jnp.zeros_like(target_states)
        max_iterations = 10
        tolerance = 0.05
        
        for iteration in range(max_iterations):
            # Calculate error
            error = target_states - current_states
            
            # Apply programming pulses where needed
            for i in range(self.rows):
                for j in range(self.cols):
                    if jnp.abs(error[i, j]) > tolerance:
                        # Determine pulse parameters
                        if error[i, j] > 0:  # Need to increase conductance (SET)
                            voltage = 3.0
                            pulse_width = 100e-9
                        else:  # Need to decrease conductance (RESET)
                            voltage = -2.0
                            pulse_width = 50e-9
                        
                        # Apply pulse
                        new_state = self.devices[i][j].write_pulse(
                            voltage, pulse_width, 'voltage'
                        )
                        current_states = current_states.at[i, j].set(new_state)
            
            # Check convergence
            if jnp.max(jnp.abs(error)) < tolerance:
                break
        
        return current_states
    
    def read_array(self, read_voltage: float = 0.1) -> chex.Array:
        """Read all device conductances."""
        conductances = jnp.zeros((self.rows, self.cols))
        
        for i in range(self.rows):
            for j in range(self.cols):
                conductance = self.devices[i][j].read_conductance(read_voltage)
                conductances = conductances.at[i, j].set(conductance)
        
        return conductances


class RRAMCrossbar:
    """RRAM crossbar array with sneak path mitigation."""
    
    def __init__(self,
                 rows: int,
                 cols: int,
                 selector_type: str = 'none',  # 'none', 'diode', '1T1R'
                 temperature: float = 300.0):
        self.rows = rows
        self.cols = cols
        self.selector_type = selector_type
        self.temperature = temperature
        
        # Initialize RRAM device array
        self.devices = []
        for i in range(rows):
            row_devices = []
            for j in range(cols):
                device = RRAMDevice(
                    oxide='HfO2',
                    thickness=5e-9,
                    area=100e-9**2,
                    temperature=temperature
                )
                row_devices.append(device)
            self.devices.append(row_devices)
    
    def __call__(self, 
                 row_voltages: chex.Array,
                 col_voltages: chex.Array,
                 params: Dict[str, Any]) -> chex.Array:
        """Compute crossbar currents with sneak path effects."""
        chex.assert_shape(row_voltages, (self.rows,))
        chex.assert_shape(col_voltages, (self.cols,))
        
        # Get device states
        states = params.get('states', jnp.ones((self.rows, self.cols)) * 0.5)
        
        # Voltage across devices
        voltage_matrix = row_voltages[:, None] - col_voltages[None, :]
        
        # Calculate device currents
        if self.selector_type == 'none':
            # No selector - direct RRAM response
            device_currents = self._calculate_rram_currents(
                voltage_matrix, states, params
            )
        elif self.selector_type == 'diode':
            # Diode selector - rectifies current
            device_currents = self._calculate_diode_selected_currents(
                voltage_matrix, states, params
            )
        elif self.selector_type == '1T1R':
            # Transistor selector - requires separate gate control
            device_currents = self._calculate_transistor_selected_currents(
                voltage_matrix, states, params
            )
        
        # Sum column currents
        col_currents = jnp.sum(device_currents, axis=0)
        
        return col_currents
    
    def _calculate_rram_currents(self,
                                voltage_matrix: chex.Array,
                                states: chex.Array,
                                params: Dict[str, Any]) -> chex.Array:
        """Calculate RRAM device currents."""
        currents = jnp.zeros_like(voltage_matrix)
        
        for i in range(self.rows):
            for j in range(self.cols):
                device_params = {
                    'state': states[i, j],
                    'is_formed': params.get('is_formed', True),
                    'include_aging': params.get('include_aging', False)
                }
                
                current = self.devices[i][j](
                    voltage_matrix[i, j], device_params
                )
                currents = currents.at[i, j].set(current)
        
        return currents
    
    def _calculate_diode_selected_currents(self,
                                         voltage_matrix: chex.Array,
                                         states: chex.Array,
                                         params: Dict[str, Any]) -> chex.Array:
        """Calculate currents with diode selectors."""
        # Diode equation: I = Is * (exp(qV/nkT) - 1)
        Is = 1e-12  # Saturation current (A)
        n = 1.5     # Ideality factor
        Vt = 0.026  # Thermal voltage at 300K
        
        # RRAM currents
        rram_currents = self._calculate_rram_currents(
            voltage_matrix, states, params
        )
        
        # Diode currents
        diode_currents = Is * (jnp.exp(voltage_matrix / (n * Vt)) - 1)
        
        # Combined current (minimum of RRAM and diode)
        # This is a simplification - actual behavior is more complex
        combined_currents = jnp.minimum(
            jnp.abs(rram_currents), 
            jnp.abs(diode_currents)
        ) * jnp.sign(rram_currents)
        
        # Diode only conducts in forward direction
        combined_currents = jnp.where(
            voltage_matrix > 0, combined_currents, 0
        )
        
        return combined_currents