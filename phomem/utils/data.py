"""
Enhanced data management system for PhoMem-CoSim.
"""

import numpy as np
import jax.numpy as jnp
import h5py
import json
import pickle
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import scipy.io as sio
from datetime import datetime
import warnings

try:
    from ..config import get_config
    from .validation import get_validator
    from .logging import get_logger
except ImportError:
    # Fallback for standalone testing
    get_config = lambda: type('Config', (), {'get_device_config': lambda: {}})() 
    get_validator = lambda: type('Validator', (), {})()
    get_logger = lambda x: type('Logger', (), {'info': print, 'error': print, 'warning': print})()

class DataManager:
    """Comprehensive data management for simulation and experimental data."""
    
    def __init__(self):
        try:
            self.config = get_config()
            self.validator = get_validator()
            self.logger = get_logger('data_manager')
        except:
            # Fallback initialization
            self.config = type('Config', (), {'get_device_config': lambda: {}})() 
            self.validator = type('Validator', (), {})()
            self.logger = type('Logger', (), {'info': print, 'error': print, 'warning': print})()
        
    def load_measurement_data(self, 
                            filepath: str,
                            data_type: str = 'auto') -> Dict[str, Any]:
        """Load experimental measurement data with automatic format detection."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        if data_type == 'auto':
            data_type = filepath.suffix.lower()
        
        try:
            if data_type in ['.h5', '.hdf5']:
                return self._load_hdf5(filepath)
            elif data_type == '.mat':
                return self._load_matlab(filepath)
            elif data_type == '.csv':
                return self._load_csv(filepath)
            elif data_type == '.json':
                return self._load_json(filepath)
            elif data_type in ['.pkl', '.pickle']:
                return self._load_pickle(filepath)
            else:
                raise ValueError(f"Unsupported data format: {data_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to load data from {filepath}: {str(e)}")
            raise
    
    def _load_hdf5(self, filepath: Path) -> Dict[str, Any]:
        """Load HDF5 measurement data."""
        data = {}
        try:
            with h5py.File(filepath, 'r') as f:
                # Device characteristics
                if 'iv_curves' in f:
                    data['iv_curves'] = {
                        'voltage': jnp.array(f['iv_curves/voltage']),
                        'current': jnp.array(f['iv_curves/current']),
                        'temperature': f['iv_curves'].attrs.get('temperature', 300.0)
                    }
                
                # Optical measurements
                if 'transmission' in f:
                    data['transmission'] = {
                        'wavelength': jnp.array(f['transmission/wavelength']),
                        'transmission': jnp.array(f['transmission/data']),
                        'phase': jnp.array(f['transmission/phase']) if 'phase' in f['transmission'] else None
                    }
                
                # Time-domain data
                if 'switching' in f:
                    data['switching_dynamics'] = {
                        'time': jnp.array(f['switching/time']),
                        'voltage': jnp.array(f['switching/voltage']),
                        'current': jnp.array(f['switching/current']),
                        'pulse_width': f['switching'].attrs.get('pulse_width', 1e-9)
                    }
                
                # Metadata
                data['metadata'] = {
                    'measurement_date': f.attrs.get('date', 'unknown'),
                    'device_id': f.attrs.get('device_id', 'unknown'),
                    'setup_config': f.attrs.get('setup_config', {})
                }
        except ImportError:
            # Fallback if h5py not available
            data = {'error': 'HDF5 support not available'}
        
        return data
    
    def _load_csv(self, filepath: Path) -> Dict[str, Any]:
        """Load CSV measurement data."""
        df = pd.read_csv(filepath)
        
        # Try to detect data type from columns
        data = {'raw_data': df}
        
        if 'voltage' in df.columns and 'current' in df.columns:
            data['iv_curves'] = {
                'voltage': jnp.array(df['voltage'].values),
                'current': jnp.array(df['current'].values),
                'temperature': df.get('temperature', pd.Series([300.0])).iloc[0]
            }
        
        if 'wavelength' in df.columns and 'transmission' in df.columns:
            data['transmission'] = {
                'wavelength': jnp.array(df['wavelength'].values),
                'transmission': jnp.array(df['transmission'].values)
            }
        
        data['metadata'] = {
            'filename': filepath.name,
            'columns': list(df.columns),
            'n_points': len(df)
        }
        
        return data
    
    def _load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load JSON measurement data."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert lists to JAX arrays where appropriate
        def convert_arrays(obj):
            if isinstance(obj, dict):
                return {k: convert_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], (int, float)):
                return jnp.array(obj)
            else:
                return obj
        
        return convert_arrays(data)
    
    def _load_matlab(self, filepath: Path) -> Dict[str, Any]:
        """Load MATLAB measurement data."""
        try:
            mat_data = sio.loadmat(filepath)
            data = {}
            
            # Convert MATLAB structures to Python dictionaries
            for key, value in mat_data.items():
                if not key.startswith('__'):
                    if isinstance(value, np.ndarray):
                        data[key] = jnp.array(value.squeeze())
                    else:
                        data[key] = value
            
            return data
        except ImportError:
            return {'error': 'MATLAB support not available'}
    
    def _load_pickle(self, filepath: Path) -> Dict[str, Any]:
        """Load pickle measurement data."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def save_simulation_results(self, 
                              results: Dict[str, Any], 
                              filepath: str,
                              format: str = 'json',
                              compress: bool = True):
        """Save simulation results with metadata."""
        filepath = Path(filepath)
        
        # Add timestamp and config metadata
        enhanced_results = {
            **results,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'phomem_version': '0.1.0',
                'config': self.config.get_device_config() if hasattr(self.config, 'get_device_config') else {},
                'simulation_params': results.get('simulation_params', {})
            }
        }
        
        try:
            if format == 'hdf5' and 'h5py' in globals():
                self._save_hdf5(enhanced_results, filepath, compress)
            elif format == 'matlab':
                self._save_matlab(enhanced_results, filepath)
            elif format == 'json':
                self._save_json(enhanced_results, filepath)
            elif format == 'pickle':
                self._save_pickle(enhanced_results, filepath)
            else:
                # Default to JSON
                self._save_json(enhanced_results, filepath.with_suffix('.json'))
        except Exception as e:
            self.logger.error(f"Failed to save in {format} format: {e}")
            # Fallback to JSON
            self._save_json(enhanced_results, filepath.with_suffix('.json'))
        
        self.logger.info(f"Results saved to {filepath}")
    
    def _save_json(self, results: Dict[str, Any], filepath: Path):
        """Save results in JSON format."""
        # Convert JAX arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, (jnp.ndarray, np.ndarray)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif hasattr(obj, 'tolist'):  # Handle other array types
                return obj.tolist()
            else:
                return obj
        
        json_data = convert_for_json(results)
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def _save_pickle(self, results: Dict[str, Any], filepath: Path):
        """Save results in pickle format."""
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    def _save_matlab(self, results: Dict[str, Any], filepath: Path):
        """Save results in MATLAB format."""
        try:
            # Convert JAX arrays to numpy for MATLAB compatibility
            def convert_for_matlab(obj):
                if isinstance(obj, jnp.ndarray):
                    return np.array(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_matlab(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_matlab(item) for item in obj]
                else:
                    return obj
            
            matlab_data = convert_for_matlab(results)
            sio.savemat(filepath, matlab_data)
        except ImportError:
            # Fallback to JSON if scipy not available
            self._save_json(results, filepath.with_suffix('.json'))

# Global instance
_data_manager = DataManager()

def load_measurement_data(filepath: str, data_type: str = 'auto') -> Dict[str, Any]:
    """Load experimental measurement data."""
    return _data_manager.load_measurement_data(filepath, data_type)

def save_simulation_results(results: Dict[str, Any], 
                          filepath: str,
                          format: str = 'json'):
    """Save simulation results."""
    return _data_manager.save_simulation_results(results, filepath, format)

def export_to_matlab(data: Dict[str, Any], filepath: str):
    """Export data to MATLAB format."""
    return _data_manager.save_simulation_results(data, filepath, 'matlab')

def import_from_csv(filepath: str) -> Dict[str, Any]:
    """Import data from CSV file."""
    return _data_manager.load_measurement_data(filepath, '.csv')