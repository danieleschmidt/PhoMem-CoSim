"""
Core multi-physics simulation engine.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import chex
from functools import partial
from abc import ABC, abstractmethod
import time

from ..photonics import MachZehnderMesh, PhotoDetectorArray
from ..memristors import PCMCrossbar, RRAMCrossbar


class PhysicsSolver(ABC):
    """Abstract base class for physics solvers."""
    
    @abstractmethod
    def solve(self, 
              geometry: Dict[str, Any],
              boundary_conditions: Dict[str, Any],
              materials: Dict[str, Any]) -> Dict[str, chex.Array]:
        """Solve physics equations for given geometry and conditions."""
        pass


class OpticalSolver(PhysicsSolver):
    """Optical field solver using various methods."""
    
    def __init__(self, 
                 method: str = 'BPM',  # 'FDTD', 'BPM', 'TMM'
                 wavelength: float = 1550e-9,
                 resolution: float = 10e-9):
        self.method = method
        self.wavelength = wavelength
        self.resolution = resolution
        self._k0 = 2 * jnp.pi / wavelength
    
    def solve(self, 
              geometry: Dict[str, Any],
              boundary_conditions: Dict[str, Any],
              materials: Dict[str, Any]) -> Dict[str, chex.Array]:
        """Solve Maxwell's equations for photonic structures."""
        
        if self.method == 'BPM':
            return self._solve_bpm(geometry, boundary_conditions, materials)
        elif self.method == 'FDTD':
            return self._solve_fdtd(geometry, boundary_conditions, materials)
        elif self.method == 'TMM':
            return self._solve_tmm(geometry, boundary_conditions, materials)
        else:
            raise ValueError(f"Unknown optical solver method: {self.method}")
    
    def _solve_bpm(self, geometry, boundary_conditions, materials):
        """Beam Propagation Method solver."""
        # Simplified BPM implementation
        nx, ny, nz = geometry['grid_size']
        dx, dy, dz = geometry['grid_spacing']
        
        # Initialize field
        field = jnp.zeros((nx, ny), dtype=jnp.complex64)
        
        # Set input field from boundary conditions
        if 'input_field' in boundary_conditions:
            field = field.at[0, :].set(boundary_conditions['input_field'])
        
        # Refractive index profile
        n_profile = self._build_index_profile(geometry, materials)
        
        # Propagate using split-step BPM
        fields = [field]
        for z_step in range(nz - 1):
            # Diffraction step (Fourier space)
            field_fft = jnp.fft.fft2(field)
            kx = jnp.fft.fftfreq(nx, dx) * 2 * jnp.pi
            ky = jnp.fft.fftfreq(ny, dy) * 2 * jnp.pi
            KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
            
            # Free space propagation
            beta = jnp.sqrt(self._k0**2 - KX**2 - KY**2 + 0j)
            phase_factor = jnp.exp(1j * beta * dz / 2)
            field_fft = field_fft * phase_factor
            field = jnp.fft.ifft2(field_fft)
            
            # Phase step (real space)
            n_slice = n_profile[:, :, z_step]
            phase_shift = jnp.exp(1j * self._k0 * (n_slice - 1.0) * dz)
            field = field * phase_shift
            
            # Second diffraction step
            field_fft = jnp.fft.fft2(field)
            field_fft = field_fft * phase_factor
            field = jnp.fft.ifft2(field_fft)
            
            fields.append(field)
        
        return {
            'electric_field': jnp.stack(fields),
            'intensity': jnp.abs(jnp.stack(fields))**2,
            'phase': jnp.angle(jnp.stack(fields))
        }
    
    def _solve_fdtd(self, geometry, boundary_conditions, materials):
        """Finite-Difference Time-Domain solver."""
        # Simplified FDTD implementation
        nx, ny, nz = geometry['grid_size']
        dx, dy, dz = geometry['grid_spacing']
        dt = geometry.get('time_step', dx / (2 * 3e8))  # Courant condition
        n_steps = geometry.get('time_steps', 1000)
        
        # Initialize fields
        Ex = jnp.zeros((nx, ny, nz))
        Ey = jnp.zeros((nx, ny, nz))
        Ez = jnp.zeros((nx, ny, nz))
        Hx = jnp.zeros((nx, ny, nz))
        Hy = jnp.zeros((nx, ny, nz))
        Hz = jnp.zeros((nx, ny, nz))
        
        # Material properties
        eps_r = self._build_permittivity_profile(geometry, materials)
        mu_r = jnp.ones_like(eps_r)  # Non-magnetic materials
        
        # FDTD coefficients
        ca = (1 - dt / (2 * eps_r * 8.854e-12)) / (1 + dt / (2 * eps_r * 8.854e-12))
        cb = dt / (eps_r * 8.854e-12 * dx) / (1 + dt / (2 * eps_r * 8.854e-12))
        
        # Time stepping (simplified)
        fields_history = []
        for step in range(n_steps):
            # Update E fields
            Ex = ca * Ex + cb * (jnp.roll(Hz, -1, axis=1) - Hz)
            Ey = ca * Ey + cb * (Hz - jnp.roll(Hz, -1, axis=0))
            
            # Update H fields
            Hz = Hz + dt / (4e-7 * jnp.pi * dx) * (
                jnp.roll(Ex, 1, axis=1) - Ex - jnp.roll(Ey, 1, axis=0) + Ey
            )
            
            # Apply source
            if 'source' in boundary_conditions:
                source_pos = boundary_conditions['source']['position']
                source_val = boundary_conditions['source']['amplitude'] * jnp.sin(
                    2 * jnp.pi * 3e8 / self.wavelength * step * dt
                )
                Ex = Ex.at[source_pos[0], source_pos[1], source_pos[2]].set(source_val)
            
            if step % 10 == 0:  # Save every 10 steps
                fields_history.append({
                    'Ex': Ex, 'Ey': Ey, 'Ez': Ez,
                    'Hx': Hx, 'Hy': Hy, 'Hz': Hz
                })
        
        return {
            'fields_history': fields_history,
            'final_fields': {'Ex': Ex, 'Ey': Ey, 'Ez': Ez, 'Hx': Hx, 'Hy': Hy, 'Hz': Hz}
        }
    
    def _build_index_profile(self, geometry, materials):
        """Build 3D refractive index profile."""
        nx, ny, nz = geometry['grid_size']
        n_profile = jnp.ones((nx, ny, nz))  # Start with vacuum
        
        # Add material regions
        for region in geometry.get('regions', []):
            material = materials[region['material']]
            n_value = material['refractive_index']
            
            # Simple box regions
            x_slice = slice(region['x_min'], region['x_max'])
            y_slice = slice(region['y_min'], region['y_max'])
            z_slice = slice(region['z_min'], region['z_max'])
            
            n_profile = n_profile.at[x_slice, y_slice, z_slice].set(n_value)
        
        return n_profile
    
    def _build_permittivity_profile(self, geometry, materials):
        """Build 3D permittivity profile."""
        n_profile = self._build_index_profile(geometry, materials)
        return n_profile**2


class ThermalSolver(PhysicsSolver):
    """Thermal diffusion solver using finite element method."""
    
    def __init__(self, method: str = 'FEM'):
        self.method = method
    
    def solve(self, 
              geometry: Dict[str, Any],
              boundary_conditions: Dict[str, Any],
              materials: Dict[str, Any]) -> Dict[str, chex.Array]:
        """Solve heat diffusion equation."""
        
        if self.method == 'FEM':
            return self._solve_fem(geometry, boundary_conditions, materials)
        else:
            return self._solve_fdm(geometry, boundary_conditions, materials)
    
    def _solve_fem(self, geometry, boundary_conditions, materials):
        """Finite Element Method thermal solver."""
        # Simplified FEM implementation using finite differences
        nx, ny, nz = geometry['grid_size']
        dx, dy, dz = geometry['grid_spacing']
        
        # Initialize temperature field
        T = jnp.ones((nx, ny, nz)) * 300.0  # 300K ambient
        
        # Material properties
        thermal_conductivity = self._build_thermal_conductivity(geometry, materials)
        heat_capacity = self._build_heat_capacity(geometry, materials)
        density = self._build_density(geometry, materials)
        
        # Heat sources from optical absorption and electrical dissipation
        heat_sources = jnp.zeros((nx, ny, nz))
        if 'heat_sources' in boundary_conditions:
            for source in boundary_conditions['heat_sources']:
                pos = source['position']
                power = source['power']
                volume = dx * dy * dz
                heat_sources = heat_sources.at[pos].add(power / volume)
        
        # Time stepping for thermal diffusion
        dt = geometry.get('thermal_time_step', 1e-6)
        n_steps = geometry.get('thermal_steps', 1000)
        
        temperatures = [T]
        for step in range(n_steps):
            # Thermal diffusion (simplified 3D diffusion)
            alpha = thermal_conductivity / (density * heat_capacity)  # Thermal diffusivity
            
            # Finite difference approximation
            d2T_dx2 = (jnp.roll(T, 1, axis=0) - 2*T + jnp.roll(T, -1, axis=0)) / dx**2
            d2T_dy2 = (jnp.roll(T, 1, axis=1) - 2*T + jnp.roll(T, -1, axis=1)) / dy**2
            d2T_dz2 = (jnp.roll(T, 1, axis=2) - 2*T + jnp.roll(T, -1, axis=2)) / dz**2
            
            dT_dt = alpha * (d2T_dx2 + d2T_dy2 + d2T_dz2) + heat_sources / (density * heat_capacity)
            
            T = T + dT_dt * dt
            
            # Apply boundary conditions
            if 'fixed_temperature' in boundary_conditions:
                for bc in boundary_conditions['fixed_temperature']:
                    pos = bc['position']
                    temp = bc['temperature']
                    T = T.at[pos].set(temp)
            
            if step % 100 == 0:  # Save every 100 steps
                temperatures.append(T)
        
        return {
            'temperature_field': jnp.stack(temperatures),
            'final_temperature': T,
            'thermal_gradient': jnp.gradient(T)
        }
    
    def _build_thermal_conductivity(self, geometry, materials):
        """Build thermal conductivity profile."""
        nx, ny, nz = geometry['grid_size']
        k_profile = jnp.ones((nx, ny, nz)) * 0.026  # Air thermal conductivity
        
        for region in geometry.get('regions', []):
            material = materials[region['material']]
            k_value = material.get('thermal_conductivity', 0.026)
            
            x_slice = slice(region['x_min'], region['x_max'])
            y_slice = slice(region['y_min'], region['y_max'])
            z_slice = slice(region['z_min'], region['z_max'])
            
            k_profile = k_profile.at[x_slice, y_slice, z_slice].set(k_value)
        
        return k_profile
    
    def _build_heat_capacity(self, geometry, materials):
        """Build heat capacity profile."""
        nx, ny, nz = geometry['grid_size']
        cp_profile = jnp.ones((nx, ny, nz)) * 1005  # Air heat capacity
        
        for region in geometry.get('regions', []):
            material = materials[region['material']]
            cp_value = material.get('heat_capacity', 1005)
            
            x_slice = slice(region['x_min'], region['x_max'])
            y_slice = slice(region['y_min'], region['y_max'])
            z_slice = slice(region['z_min'], region['z_max'])
            
            cp_profile = cp_profile.at[x_slice, y_slice, z_slice].set(cp_value)
        
        return cp_profile
    
    def _build_density(self, geometry, materials):
        """Build density profile."""
        nx, ny, nz = geometry['grid_size']
        rho_profile = jnp.ones((nx, ny, nz)) * 1.225  # Air density
        
        for region in geometry.get('regions', []):
            material = materials[region['material']]
            rho_value = material.get('density', 1.225)
            
            x_slice = slice(region['x_min'], region['x_max'])
            y_slice = slice(region['y_min'], region['y_max'])
            z_slice = slice(region['z_min'], region['z_max'])
            
            rho_profile = rho_profile.at[x_slice, y_slice, z_slice].set(rho_value)
        
        return rho_profile


class ElectricalSolver(PhysicsSolver):
    """Electrical circuit solver using modified nodal analysis."""
    
    def __init__(self, method: str = 'SPICE'):
        self.method = method
    
    def solve(self, 
              geometry: Dict[str, Any],
              boundary_conditions: Dict[str, Any],
              materials: Dict[str, Any]) -> Dict[str, chex.Array]:
        """Solve electrical circuit equations."""
        
        if self.method == 'SPICE':
            return self._solve_spice_like(geometry, boundary_conditions, materials)
        else:
            return self._solve_nodal_analysis(geometry, boundary_conditions, materials)
    
    def _solve_spice_like(self, geometry, boundary_conditions, materials):
        """SPICE-like circuit simulation."""
        # Circuit netlist parsing and solution
        nodes = geometry.get('nodes', [])
        elements = geometry.get('elements', [])
        n_nodes = len(nodes)
        
        # Build admittance matrix
        Y = jnp.zeros((n_nodes, n_nodes), dtype=jnp.complex64)
        I = jnp.zeros(n_nodes, dtype=jnp.complex64)
        
        for element in elements:
            if element['type'] == 'resistor':
                n1, n2 = element['nodes']
                R = element['value']
                G = 1.0 / R
                
                Y = Y.at[n1, n1].add(G)
                Y = Y.at[n2, n2].add(G)
                Y = Y.at[n1, n2].add(-G)
                Y = Y.at[n2, n1].add(-G)
            
            elif element['type'] == 'current_source':
                node = element['node']
                current = element['value']
                I = I.at[node].add(current)
            
            elif element['type'] == 'voltage_source':
                # Voltage sources require modified nodal analysis
                # Simplified: convert to Norton equivalent
                node = element['node']
                voltage = element['value']
                R_series = element.get('series_resistance', 1e-6)
                I = I.at[node].add(voltage / R_series)
                Y = Y.at[node, node].add(1.0 / R_series)
        
        # Solve Y * V = I
        # Handle singular matrix (ground node)
        if n_nodes > 1:
            Y_reduced = Y[1:, 1:]  # Remove ground node
            I_reduced = I[1:]
            
            V_reduced = jnp.linalg.solve(Y_reduced, I_reduced)
            V = jnp.concatenate([jnp.array([0.0]), V_reduced])  # Ground = 0V
        else:
            V = jnp.array([0.0])
        
        # Calculate currents through elements
        currents = {}
        for i, element in enumerate(elements):
            if element['type'] == 'resistor':
                n1, n2 = element['nodes']
                R = element['value']
                current = (V[n1] - V[n2]) / R
                currents[f"R{i}"] = current
        
        return {
            'node_voltages': V,
            'element_currents': currents,
            'power_dissipation': sum([jnp.real(V[elem['nodes'][0]] * jnp.conj(currents.get(f"R{i}", 0))) 
                                    for i, elem in enumerate(elements) if elem['type'] == 'resistor'])
        }


class MultiPhysicsSimulator:
    """Main multi-physics co-simulation engine."""
    
    def __init__(self,
                 optical_solver: str = 'BPM',
                 thermal_solver: str = 'FEM',
                 electrical_solver: str = 'SPICE',
                 coupling: str = 'weak'):  # 'weak', 'strong'
        
        self.optical_solver = OpticalSolver(method=optical_solver)
        self.thermal_solver = ThermalSolver(method=thermal_solver)
        self.electrical_solver = ElectricalSolver(method=electrical_solver)
        self.coupling = coupling
        
        # Coupling parameters
        self.max_iterations = 10 if coupling == 'strong' else 1
        self.convergence_tolerance = 1e-6
    
    def simulate(self,
                chip_design: Any,
                input_optical_power: float = 10e-3,
                ambient_temperature: float = 25,
                duration: float = 1.0,
                save_fields: bool = False) -> Dict[str, Any]:
        """
        Run multi-physics co-simulation.
        
        Args:
            chip_design: ChipDesign object with geometry and materials
            input_optical_power: Input optical power (W)
            ambient_temperature: Ambient temperature (°C)
            duration: Simulation duration (s)
            save_fields: Whether to save field distributions
            
        Returns:
            Simulation results dictionary
        """
        start_time = time.time()
        
        # Extract simulation domains from chip design
        geometry = chip_design.get_geometry()
        materials = chip_design.get_materials()
        
        # Initialize boundary conditions
        optical_bc = {
            'input_field': jnp.sqrt(input_optical_power) * jnp.ones(10),  # Simplified
        }
        
        thermal_bc = {
            'fixed_temperature': [{'position': (0, 0, 0), 'temperature': ambient_temperature + 273.15}],
            'heat_sources': []
        }
        
        electrical_bc = {
            'voltage_sources': [],
            'current_sources': []
        }
        
        # Coupling iteration loop
        optical_results = None
        thermal_results = None
        electrical_results = None
        
        for iteration in range(self.max_iterations):
            print(f"Coupling iteration {iteration + 1}/{self.max_iterations}")
            
            # Optical simulation
            optical_results = self.optical_solver.solve(geometry, optical_bc, materials)
            
            # Extract heat sources from optical absorption
            if 'intensity' in optical_results:
                absorption_coeff = 0.1  # 1/cm (simplified)
                heat_generation = optical_results['intensity'] * absorption_coeff
                thermal_bc['heat_sources'] = [
                    {'position': (i, j, k), 'power': heat_generation[i, j, k].item()}
                    for i in range(heat_generation.shape[0])
                    for j in range(heat_generation.shape[1])
                    for k in range(heat_generation.shape[2])
                    if heat_generation[i, j, k] > 1e-6  # Threshold
                ]
            
            # Thermal simulation
            thermal_results = self.thermal_solver.solve(geometry, thermal_bc, materials)
            
            # Electrical simulation with temperature-dependent resistances
            electrical_results = self.electrical_solver.solve(geometry, electrical_bc, materials)
            
            # Check convergence for strong coupling
            if self.coupling == 'strong' and iteration > 0:
                # Simplified convergence check
                temp_change = jnp.max(jnp.abs(
                    thermal_results['final_temperature'] - prev_temp
                ))
                if temp_change < self.convergence_tolerance:
                    print(f"Converged after {iteration + 1} iterations")
                    break
            
            prev_temp = thermal_results['final_temperature']
        
        # Compile results
        simulation_time = time.time() - start_time
        
        results = {
            'optical': optical_results,
            'thermal': thermal_results,
            'electrical': electrical_results,
            'simulation_time': simulation_time,
            'converged': iteration < self.max_iterations - 1 if self.coupling == 'strong' else True
        }
        
        if not save_fields:
            # Remove large field arrays to save memory
            if 'fields_history' in results['optical']:
                del results['optical']['fields_history']
            if 'temperature_field' in results['thermal']:
                results['thermal']['temperature_field'] = results['thermal']['temperature_field'][-1:]  # Keep only final
        
        return results
    
    def plot_thermal_map(self, 
                        temperature_field: chex.Array,
                        save_path: Optional[str] = None) -> None:
        """Plot 2D thermal distribution."""
        try:
            import matplotlib.pyplot as plt
            
            # Take middle z-slice for 2D visualization
            if temperature_field.ndim == 3:
                temp_2d = temperature_field[:, :, temperature_field.shape[2]//2]
            else:
                temp_2d = temperature_field
            
            plt.figure(figsize=(10, 8))
            im = plt.imshow(temp_2d.T, cmap='hot', origin='lower')
            plt.colorbar(im, label='Temperature (K)')
            plt.title('Temperature Distribution')
            plt.xlabel('X position')
            plt.ylabel('Y position')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")