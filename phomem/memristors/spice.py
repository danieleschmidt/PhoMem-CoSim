"""
SPICE integration for circuit-level simulation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import subprocess
import tempfile
import os
from pathlib import Path


class SPICEInterface:
    """Interface to ngspice for circuit simulation."""
    
    def __init__(self, 
                 spice_executable: str = 'ngspice',
                 temp_dir: Optional[str] = None):
        self.spice_executable = spice_executable
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
        # Verify ngspice is available
        try:
            result = subprocess.run([spice_executable, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                raise RuntimeError(f"ngspice not found or not working: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"ngspice not available: {e}")
    
    def run_simulation(self, 
                      netlist: str,
                      analysis_type: str = 'dc',
                      parameters: Dict[str, Any] = None) -> Dict[str, np.ndarray]:
        """Run SPICE simulation and return results."""
        parameters = parameters or {}
        
        # Create temporary netlist file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', 
                                       dir=self.temp_dir, delete=False) as f:
            f.write(netlist)
            netlist_path = f.name
        
        try:
            # Run ngspice
            cmd = [self.spice_executable, '-b', netlist_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise RuntimeError(f"SPICE simulation failed: {result.stderr}")
            
            # Parse results
            results = self._parse_spice_output(result.stdout, analysis_type)
            
        finally:
            # Clean up temporary file
            os.unlink(netlist_path)
        
        return results
    
    def _parse_spice_output(self, 
                           output: str,
                           analysis_type: str) -> Dict[str, np.ndarray]:
        """Parse SPICE output into structured data."""
        results = {}
        lines = output.split('\n')
        
        if analysis_type == 'dc':
            # Parse DC analysis results
            data_started = False
            data_lines = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('Index'):  # Header line
                    headers = line.split()
                    data_started = True
                    continue
                elif data_started and line:
                    try:
                        values = [float(x) for x in line.split()]
                        data_lines.append(values)
                    except ValueError:
                        continue
            
            if data_lines:
                data_array = np.array(data_lines)
                for i, header in enumerate(headers):
                    if i < data_array.shape[1]:
                        results[header] = data_array[:, i]
        
        elif analysis_type == 'tran':
            # Parse transient analysis results
            # Similar parsing logic for time-domain data
            pass
        
        return results


def generate_spice_netlist(network, 
                         include_parasitics: bool = True,
                         corner: str = 'tt_25c',
                         supply_voltage: float = 3.3) -> str:
    """Generate SPICE netlist from network description."""
    
    netlist_lines = [
        f"* PhoMem-CoSim Generated Netlist",
        f"* Process corner: {corner}",
        f"* Supply voltage: {supply_voltage}V",
        f"",
        f".title PhoMem Hybrid Network Simulation",
        f"",
    ]
    
    # Add supply and reference
    netlist_lines.extend([
        f"VDD VDD 0 DC {supply_voltage}",
        f"VSS VSS 0 DC 0",
        f"",
    ])
    
    # Add memristor models
    netlist_lines.extend(_generate_memristor_models())
    
    # Add photodetector models
    netlist_lines.extend(_generate_photodetector_models())
    
    # Add circuit elements based on network structure
    if hasattr(network, 'memristor_layers'):
        for i, layer in enumerate(network.memristor_layers):
            netlist_lines.extend(
                _generate_crossbar_netlist(layer, f"X{i}")
            )
    
    # Add analysis commands
    netlist_lines.extend([
        f"",
        f".control",
        f"dc VDD 0 {supply_voltage} 0.1",
        f"print all",
        f".endc",
        f"",
        f".end"
    ])
    
    return '\n'.join(netlist_lines)


def _generate_memristor_models() -> List[str]:
    """Generate SPICE models for memristive devices."""
    models = [
        "* PCM Device Model",
        ".model PCM_GST225 memristor (",
        "+ Ron=1k Roff=1Meg Rinit=100k",
        "+ Vth=1.5 pon=1 poff=1",
        "+ model=1 )",
        "",
        "* RRAM Device Model", 
        ".model RRAM_HfO2 memristor (",
        "+ Ron=5k Roff=1Meg Rinit=500k",
        "+ Vth=1.2 pon=1 poff=1",
        "+ model=1 )",
        "",
    ]
    return models


def _generate_photodetector_models() -> List[str]:
    """Generate SPICE models for photodetectors."""
    models = [
        "* Photodetector Model",
        ".subckt PHOTODETECTOR ANODE CATHODE OPTICAL_IN",
        "* Responsivity = 0.8 A/W",
        "* Dark current = 1nA", 
        "GPHOTO ANODE CATHODE OPTICAL_IN 0 0.8",
        "IDARK CATHODE ANODE DC 1n",
        "CJUNCTION ANODE CATHODE 10f",
        ".ends PHOTODETECTOR",
        "",
    ]
    return models


def _generate_crossbar_netlist(crossbar, instance_name: str) -> List[str]:
    """Generate SPICE netlist for memristor crossbar."""
    netlist = [
        f"* Crossbar Array: {instance_name}",
        f"* Size: {crossbar.rows}x{crossbar.cols}",
        "",
    ]
    
    # Generate memristor instances
    for i in range(crossbar.rows):
        for j in range(crossbar.cols):
            device_name = f"M{instance_name}_{i}_{j}"
            row_node = f"ROW_{i}"
            col_node = f"COL_{j}"
            
            if hasattr(crossbar, 'devices') and isinstance(crossbar.devices[0][0], type):
                # PCM devices
                device_type = "PCM_GST225"
            else:
                # RRAM devices
                device_type = "RRAM_HfO2"
            
            netlist.append(
                f"{device_name} {row_node} {col_node} {device_type}"
            )
    
    netlist.append("")
    return netlist


def run_corner_simulations(netlist: str,
                          testbench: str,
                          corners: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Run corner analysis across process/voltage/temperature variations."""
    
    spice = SPICEInterface()
    results = {}
    
    for corner in corners:
        # Modify netlist for corner conditions
        corner_netlist = _modify_netlist_for_corner(netlist, corner)
        
        # Add testbench
        if testbench:
            with open(testbench, 'r') as f:
                tb_content = f.read()
            corner_netlist += "\n" + tb_content
        
        # Run simulation
        try:
            corner_results = spice.run_simulation(corner_netlist, 'dc')
            results[corner] = corner_results
        except RuntimeError as e:
            print(f"Warning: Corner {corner} simulation failed: {e}")
            results[corner] = {}
    
    return results


def _modify_netlist_for_corner(netlist: str, corner: str) -> str:
    """Modify netlist parameters for specific corner."""
    
    corner_params = {
        'ff_-40c': {'temp': -40, 'process': 'fast', 'voltage': 1.1},
        'tt_25c': {'temp': 25, 'process': 'typical', 'voltage': 1.0},
        'ss_125c': {'temp': 125, 'process': 'slow', 'voltage': 0.9}
    }
    
    if corner not in corner_params:
        return netlist
    
    params = corner_params[corner]
    
    # Add temperature setting
    temp_line = f".temp {params['temp']}"
    
    # Modify supply voltage
    voltage_multiplier = params['voltage']
    modified_netlist = netlist.replace(
        "VDD VDD 0 DC 3.3",
        f"VDD VDD 0 DC {3.3 * voltage_multiplier}"
    )
    
    # Add process variations (simplified)
    if params['process'] == 'fast':
        # Reduce resistance values by 20%
        modified_netlist = modified_netlist.replace("Ron=1k", "Ron=800")
        modified_netlist = modified_netlist.replace("Ron=5k", "Ron=4k")
    elif params['process'] == 'slow':
        # Increase resistance values by 20%
        modified_netlist = modified_netlist.replace("Ron=1k", "Ron=1.2k")
        modified_netlist = modified_netlist.replace("Ron=5k", "Ron=6k")
    
    # Add temperature line
    modified_netlist = modified_netlist.replace(
        ".title PhoMem Hybrid Network Simulation",
        f".title PhoMem Hybrid Network Simulation\n{temp_line}"
    )
    
    return modified_netlist


class MemristorSPICEModel:
    """Behavioral SPICE model for memristors with state evolution."""
    
    def __init__(self, device_type: str = 'PCM'):
        self.device_type = device_type
        self.state_file = None
    
    def generate_verilog_a_model(self) -> str:
        """Generate Verilog-A model for advanced SPICE simulators."""
        
        if self.device_type == 'PCM':
            return self._generate_pcm_verilog_a()
        elif self.device_type == 'RRAM':
            return self._generate_rram_verilog_a()
        else:
            raise ValueError(f"Unknown device type: {self.device_type}")
    
    def _generate_pcm_verilog_a(self) -> str:
        """Generate PCM Verilog-A model."""
        return '''
`include "constants.vams"
`include "disciplines.vams"

module pcm_device(te, be);
    inout te, be;
    electrical te, be;
    
    parameter real ron = 1e3;        // ON resistance (Ohm)
    parameter real roff = 1e6;       // OFF resistance (Ohm)
    parameter real vset = 3.0;       // SET voltage (V)
    parameter real vreset = 1.5;     // RESET voltage (V)
    parameter real tset = 100e-9;    // SET time (s)
    parameter real treset = 50e-9;   // RESET time (s)
    
    real conductance, voltage, current;
    real state;  // 0=amorphous, 1=crystalline
    
    analog begin
        voltage = V(te, be);
        
        // State evolution
        if (abs(voltage) > vset && state < 1.0) begin
            state = state + dt/tset;
            if (state > 1.0) state = 1.0;
        end else if (abs(voltage) > vreset && state > 0.0) begin
            state = state - dt/treset;
            if (state < 0.0) state = 0.0;
        end
        
        // Conductance calculation
        conductance = 1.0/roff + state * (1.0/ron - 1.0/roff);
        
        // Current calculation
        current = conductance * voltage;
        I(te, be) <+ current;
    end
    
endmodule
'''
    
    def _generate_rram_verilog_a(self) -> str:
        """Generate RRAM Verilog-A model."""
        return '''
`include "constants.vams"
`include "disciplines.vams"

module rram_device(te, be);
    inout te, be;
    electrical te, be;
    
    parameter real ron = 5e3;        // ON resistance (Ohm)
    parameter real roff = 1e6;       // OFF resistance (Ohm)
    parameter real vset = 1.2;       // SET voltage (V)
    parameter real vreset = -1.0;    // RESET voltage (V)
    parameter real tswitch = 1e-9;   // Switching time (s)
    
    real conductance, voltage, current;
    real state;  // 0=HRS, 1=LRS
    
    analog begin
        voltage = V(te, be);
        
        // Switching dynamics
        if (voltage > vset && state < 1.0) begin
            state = state + dt/tswitch;
            if (state > 1.0) state = 1.0;
        end else if (voltage < vreset && state > 0.0) begin
            state = state - dt/tswitch;
            if (state < 0.0) state = 0.0;
        end
        
        // Conductance calculation
        conductance = 1.0/roff + state * (1.0/ron - 1.0/roff);
        
        // Current with nonlinearity
        current = conductance * voltage * pow(abs(voltage), 0.2);
        I(te, be) <+ current;
    end
    
endmodule
'''