"""
Multi-physics co-simulation components and chip design utilities.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import chex
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .core import OpticalSolver, ThermalSolver, ElectricalSolver


@dataclass
class MaterialProperties:
    """Material properties for multi-physics simulation."""
    name: str
    refractive_index: complex = 1.0 + 0j
    thermal_conductivity: float = 1.0  # W/m·K
    heat_capacity: float = 1000.0  # J/kg·K
    density: float = 1000.0  # kg/m³
    electrical_conductivity: float = 1e-6  # S/m
    
    # Optical properties
    absorption_coefficient: float = 0.0  # 1/m
    scattering_coefficient: float = 0.0  # 1/m
    
    # Thermal properties
    thermal_expansion: float = 1e-6  # 1/K
    
    # Electrical properties
    permittivity: float = 1.0
    permeability: float = 1.0


class ChipDesign:
    """Chip design specification for multi-physics simulation."""
    
    def __init__(self, name: str = "PhoMem_Chip"):
        self.name = name
        self.photonic_layers = []
        self.electronic_layers = []
        self.thermal_interfaces = []
        self.materials = {}
        self.geometry = {}
        
        # Add default materials
        self._add_default_materials()
    
    def _add_default_materials(self):
        """Add default material library."""
        self.materials.update({
            'silicon': MaterialProperties(
                name='silicon',
                refractive_index=3.45 + 0.01j,
                thermal_conductivity=150.0,
                heat_capacity=700.0,
                density=2330.0,
                electrical_conductivity=1e-4,
                absorption_coefficient=100.0
            ),
            'silicon_nitride': MaterialProperties(
                name='silicon_nitride',
                refractive_index=2.0 + 0.001j,
                thermal_conductivity=30.0,
                heat_capacity=800.0,
                density=3100.0,
                absorption_coefficient=0.1
            ),
            'sio2': MaterialProperties(
                name='sio2',
                refractive_index=1.45 + 0j,
                thermal_conductivity=1.4,
                heat_capacity=1000.0,
                density=2200.0,
                absorption_coefficient=0.01
            ),
            'gst225': MaterialProperties(
                name='gst225',
                refractive_index=4.0 + 0.05j,  # Amorphous
                thermal_conductivity=0.5,
                heat_capacity=200.0,
                density=6000.0,
                electrical_conductivity=1e-3,
                absorption_coefficient=1000.0
            ),
            'copper': MaterialProperties(
                name='copper',
                refractive_index=0.2 + 5.0j,
                thermal_conductivity=400.0,
                heat_capacity=385.0,
                density=8960.0,
                electrical_conductivity=5.8e7
            ),
            'air': MaterialProperties(
                name='air',
                refractive_index=1.0 + 0j,
                thermal_conductivity=0.026,
                heat_capacity=1005.0,
                density=1.225
            )
        })
    
    def add_photonic_die(self, photonic_components: List[Any]):
        """Add photonic die with components."""
        self.photonic_layers.extend(photonic_components)
    
    def add_electronic_die(self, electronic_components: List[Any]):
        """Add electronic die with components."""
        self.electronic_layers.extend(electronic_components)
    
    def add_thermal_interface(self, 
                             material: str = 'diamond',
                             thickness: float = 100e-6):
        """Add thermal interface material."""
        interface = {
            'material': material,
            'thickness': thickness,
            'thermal_conductivity': 2000.0 if material == 'diamond' else 400.0
        }
        self.thermal_interfaces.append(interface)
    
    def set_geometry(self, 
                    grid_size: Tuple[int, int, int],
                    physical_size: Tuple[float, float, float],
                    regions: List[Dict[str, Any]] = None):
        """Set chip geometry and discretization."""
        self.geometry = {
            'grid_size': grid_size,
            'physical_size': physical_size,
            'grid_spacing': tuple(p/g for p, g in zip(physical_size, grid_size)),
            'regions': regions or []
        }
    
    def get_geometry(self) -> Dict[str, Any]:
        """Get geometry specification."""
        if not self.geometry:
            # Default geometry
            self.set_geometry(
                grid_size=(100, 100, 20),
                physical_size=(1000e-6, 1000e-6, 200e-6),  # 1mm x 1mm x 200μm
                regions=[
                    {
                        'name': 'photonic_layer',
                        'material': 'silicon',
                        'x_min': 20, 'x_max': 80,
                        'y_min': 20, 'y_max': 80,
                        'z_min': 5, 'z_max': 10
                    },
                    {
                        'name': 'electronic_layer',
                        'material': 'silicon',
                        'x_min': 20, 'x_max': 80,
                        'y_min': 20, 'y_max': 80,
                        'z_min': 10, 'z_max': 15
                    }
                ]
            )
        return self.geometry
    
    def get_materials(self) -> Dict[str, Dict[str, Any]]:
        """Get materials dictionary for solvers."""
        materials_dict = {}
        for name, props in self.materials.items():
            materials_dict[name] = {
                'refractive_index': props.refractive_index,
                'thermal_conductivity': props.thermal_conductivity,
                'heat_capacity': props.heat_capacity,
                'density': props.density,
                'electrical_conductivity': props.electrical_conductivity,
                'absorption_coefficient': props.absorption_coefficient
            }
        return materials_dict


class FieldSolver:
    """Electromagnetic field solver for photonic components."""
    
    def __init__(self, method: str = 'FEM'):
        self.method = method
    
    def solve_waveguide_modes(self,
                             geometry: Dict[str, Any],
                             wavelength: float = 1550e-9,
                             n_modes: int = 1) -> Dict[str, chex.Array]:
        """Solve for waveguide modes."""
        # Simplified mode solver
        k0 = 2 * jnp.pi / wavelength
        
        # Effective index calculation (simplified)
        n_core = 3.45  # Silicon
        n_clad = 1.45  # SiO2
        
        # Slab waveguide approximation
        thickness = geometry.get('thickness', 220e-9)
        width = geometry.get('width', 450e-9)
        
        # Effective index using slab waveguide formula
        V = k0 * thickness * jnp.sqrt(n_core**2 - n_clad**2)  # V-parameter
        
        if V > jnp.pi/2:  # Single mode condition
            n_eff = n_clad + (n_core - n_clad) * (1 - (jnp.pi/(2*V))**2)
        else:
            n_eff = n_clad
        
        # Gaussian mode profile (approximation)
        x = jnp.linspace(-width, width, 100)
        y = jnp.linspace(-thickness, thickness, 50)
        X, Y = jnp.meshgrid(x, y)
        
        # Mode field diameter
        w0 = wavelength / (jnp.pi * jnp.sqrt(n_core**2 - n_eff**2))
        
        # Fundamental mode
        mode_field = jnp.exp(-(X**2 + Y**2) / w0**2)
        
        return {
            'effective_index': n_eff,
            'mode_field': mode_field,
            'propagation_constant': k0 * n_eff,
            'coordinates': (X, Y)
        }
    
    def couple_modes(self,
                    mode1: Dict[str, chex.Array],
                    mode2: Dict[str, chex.Array],
                    separation: float) -> float:
        """Calculate coupling coefficient between modes."""
        # Overlap integral for coupling
        field1 = mode1['mode_field']
        field2 = mode2['mode_field']
        
        # Normalize modes
        norm1 = jnp.sqrt(jnp.sum(jnp.abs(field1)**2))
        norm2 = jnp.sqrt(jnp.sum(jnp.abs(field2)**2))
        
        field1_norm = field1 / norm1
        field2_norm = field2 / norm2
        
        # Coupling coefficient (simplified)
        overlap = jnp.sum(field1_norm * jnp.conj(field2_norm))
        coupling = jnp.abs(overlap) * jnp.exp(-separation / 1e-6)  # Exponential decay
        
        return coupling


class CoupledSimulation:
    """Manages coupled multi-physics simulations."""
    
    def __init__(self, 
                 optical_solver: OpticalSolver,
                 thermal_solver: ThermalSolver,
                 electrical_solver: ElectricalSolver):
        self.optical_solver = optical_solver
        self.thermal_solver = thermal_solver
        self.electrical_solver = electrical_solver
        
        # Coupling matrices
        self.opto_thermal_coupling = None
        self.thermo_optical_coupling = None
        self.electro_thermal_coupling = None
    
    def setup_coupling(self, geometry: Dict[str, Any]):
        """Setup coupling matrices between physics domains."""
        nx, ny, nz = geometry['grid_size']
        n_total = nx * ny * nz
        
        # Opto-thermal: optical absorption -> heat generation
        self.opto_thermal_coupling = jnp.eye(n_total) * 0.1  # 10% absorption
        
        # Thermo-optical: temperature -> refractive index change
        dn_dT = 1.86e-4  # /K for silicon
        self.thermo_optical_coupling = jnp.eye(n_total) * dn_dT
        
        # Electro-thermal: Joule heating
        self.electro_thermal_coupling = jnp.eye(n_total) * 1.0
    
    def solve_coupled_system(self,
                           geometry: Dict[str, Any],
                           boundary_conditions: Dict[str, Any],
                           materials: Dict[str, Any],
                           max_iterations: int = 10,
                           tolerance: float = 1e-6) -> Dict[str, Any]:
        """Solve coupled multi-physics system iteratively."""
        
        # Initialize fields
        optical_fields = None
        temperature_field = jnp.ones(geometry['grid_size']) * 300.0  # 300K
        electrical_fields = None
        
        # Setup coupling
        self.setup_coupling(geometry)
        
        # Iteration loop
        for iteration in range(max_iterations):
            print(f"Coupled iteration {iteration + 1}")
            
            # Update optical boundary conditions with temperature effects
            optical_bc = boundary_conditions.get('optical', {})
            if self.thermo_optical_coupling is not None and optical_fields is not None:
                # Temperature-dependent refractive index
                delta_n = jnp.mean(temperature_field - 300.0) * 1.86e-4
                # Update material properties
                for mat_name, mat_props in materials.items():
                    if 'refractive_index' in mat_props:
                        mat_props['refractive_index'] += delta_n
            
            # Solve optical fields
            optical_results = self.optical_solver.solve(geometry, optical_bc, materials)
            optical_fields = optical_results.get('electric_field', optical_fields)
            
            # Update thermal boundary conditions with optical heating
            thermal_bc = boundary_conditions.get('thermal', {})
            if optical_fields is not None:
                # Calculate absorption heating
                intensity = jnp.abs(optical_fields)**2
                absorption_heating = intensity * 0.1  # 10% absorption
                
                # Add to heat sources
                if 'heat_sources' not in thermal_bc:
                    thermal_bc['heat_sources'] = []
                
                # Convert to point sources (simplified)
                avg_heating = jnp.mean(absorption_heating)
                if avg_heating > 1e-6:
                    thermal_bc['heat_sources'].append({
                        'position': (geometry['grid_size'][0]//2, 
                                   geometry['grid_size'][1]//2,
                                   geometry['grid_size'][2]//2),
                        'power': avg_heating
                    })
            
            # Solve thermal fields
            thermal_results = self.thermal_solver.solve(geometry, thermal_bc, materials)
            new_temperature_field = thermal_results['final_temperature']
            
            # Check convergence
            if iteration > 0:
                temp_change = jnp.max(jnp.abs(new_temperature_field - temperature_field))
                print(f"  Temperature change: {temp_change:.2e} K")
                if temp_change < tolerance:
                    print(f"Converged after {iteration + 1} iterations")
                    break
            
            temperature_field = new_temperature_field
            
            # Solve electrical fields (if needed)
            electrical_bc = boundary_conditions.get('electrical', {})
            electrical_results = self.electrical_solver.solve(geometry, electrical_bc, materials)
            electrical_fields = electrical_results
        
        return {
            'optical': optical_results,
            'thermal': thermal_results,
            'electrical': electrical_results,
            'converged': iteration < max_iterations - 1,
            'iterations': iteration + 1
        }


def export_to_gds(photonic_layers: List[Any],
                 foundry: str = 'imec_sin',
                 design_rules: str = 'drc_2025.tech') -> str:
    """Export photonic layout to GDS format."""
    
    # This would interface with actual GDS libraries like gdspy or gdstk
    # For now, return a mock GDS string
    
    gds_content = f"""
/* PhoMem-CoSim Generated GDS Layout */
/* Foundry: {foundry} */
/* Design Rules: {design_rules} */

HEADER 600
BGNLIB 
  LASTMOD {{2025,1,1,12,0,0}}
  LASTACC {{2025,1,1,12,0,0}}
LIBNAME PHOMEM_CHIP.DB

BGNSTR
  CREATION {{2025,1,1,12,0,0}}
  LASTMOD {{2025,1,1,12,0,0}}
STRNAME PHOMEM_TOP

/* Waveguide layer (layer 1) */
BOUNDARY
LAYER 1
DATATYPE 0
XY
  {' '.join([str(int(x*1e6)) for x in [0, 0, 1000, 0, 1000, 100, 0, 100, 0, 0]])}
ENDEL

/* Phase shifter regions (layer 2) */
BOUNDARY
LAYER 2
DATATYPE 0
XY
  {' '.join([str(int(x*1e6)) for x in [200, 20, 800, 20, 800, 80, 200, 80, 200, 20]])}
ENDEL

ENDSTR
ENDLIB
"""
    
    return gds_content


def export_crossbar_layout(memristor_layers: List[Any],
                          technology: str = '28nm_cmos',
                          metal_layers: List[str] = ['M4', 'M5']) -> Dict[str, Any]:
    """Export memristor crossbar layout."""
    
    layout_data = {
        'technology': technology,
        'metal_layers': metal_layers,
        'devices': [],
        'interconnects': []
    }
    
    for i, layer in enumerate(memristor_layers):
        if hasattr(layer, 'rows') and hasattr(layer, 'cols'):
            rows, cols = layer.rows, layer.cols
            
            # Generate device positions
            device_pitch = 100e-9  # 100nm pitch
            
            for r in range(rows):
                for c in range(cols):
                    device = {
                        'type': 'memristor',
                        'position': (c * device_pitch, r * device_pitch),
                        'size': (50e-9, 50e-9),  # 50nm x 50nm
                        'layer': f'layer_{i}',
                        'row': r,
                        'col': c
                    }
                    layout_data['devices'].append(device)
            
            # Add word lines (rows)
            for r in range(rows):
                wordline = {
                    'type': 'wordline',
                    'layer': metal_layers[0],
                    'start': (0, r * device_pitch),
                    'end': (cols * device_pitch, r * device_pitch),
                    'width': 20e-9
                }
                layout_data['interconnects'].append(wordline)
            
            # Add bit lines (columns)
            for c in range(cols):
                bitline = {
                    'type': 'bitline',
                    'layer': metal_layers[1],
                    'start': (c * device_pitch, 0),
                    'end': (c * device_pitch, rows * device_pitch),
                    'width': 20e-9
                }
                layout_data['interconnects'].append(bitline)
    
    return layout_data