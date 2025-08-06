"""
Device calibration system for accurate hardware modeling.
"""

import logging
import json
import numpy as np
import scipy.optimize as opt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import warnings

import jax
import jax.numpy as jnp
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """Container for device calibration measurements."""
    device_type: str
    material: str
    measurements: Dict[str, np.ndarray]
    conditions: Dict[str, Any]
    timestamp: str
    source: str = "experiment"
    
    def validate(self) -> bool:
        """Validate calibration data integrity."""
        required_keys = {
            'pcm_mushroom': ['voltage', 'current', 'resistance', 'temperature'],
            'rram_hfo2': ['voltage', 'current', 'set_voltage', 'reset_voltage'],
            'thermal_phase_shifter': ['power', 'phase_shift', 'response_time'],
            'plasma_phase_shifter': ['voltage', 'phase_shift', 'current']
        }
        
        if self.device_type not in required_keys:
            logger.error(f"Unknown device type: {self.device_type}")
            return False
        
        missing_keys = set(required_keys[self.device_type]) - set(self.measurements.keys())
        if missing_keys:
            logger.error(f"Missing measurement keys: {missing_keys}")
            return False
        
        # Check array lengths match
        array_lengths = [len(arr) for arr in self.measurements.values()]
        if len(set(array_lengths)) > 1:
            logger.error("Measurement arrays have different lengths")
            return False
        
        logger.info(f"Calibration data validated for {self.device_type}")
        return True
    
    def save(self, path: Union[str, Path]):
        """Save calibration data to file."""
        path = Path(path)
        data = {
            'device_type': self.device_type,
            'material': self.material,
            'measurements': {k: v.tolist() for k, v in self.measurements.items()},
            'conditions': self.conditions,
            'timestamp': self.timestamp,
            'source': self.source
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Calibration data saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'CalibrationData':
        """Load calibration data from file."""
        path = Path(path)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to arrays
        measurements = {k: np.array(v) for k, v in data['measurements'].items()}
        
        return cls(
            device_type=data['device_type'],
            material=data['material'],
            measurements=measurements,
            conditions=data['conditions'],
            timestamp=data['timestamp'],
            source=data.get('source', 'experiment')
        )


class DeviceCalibrator:
    """Base class for device calibration."""
    
    def __init__(self, device_type: str):
        self.device_type = device_type
        self.calibrated_parameters = {}
        self.calibration_quality = {}
        
    def calibrate(self, data: CalibrationData) -> Dict[str, Any]:
        """Calibrate device model parameters."""
        if not data.validate():
            raise ValueError("Invalid calibration data")
        
        if data.device_type != self.device_type:
            raise ValueError(f"Data type {data.device_type} doesn't match calibrator {self.device_type}")
        
        # Override in subclasses
        raise NotImplementedError("Subclasses must implement calibrate method")
    
    def validate_calibration(self, data: CalibrationData, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Validate calibration quality using test data."""
        # Override in subclasses
        raise NotImplementedError("Subclasses must implement validate_calibration method")
    
    def save_calibration(self, path: Union[str, Path]):
        """Save calibrated parameters."""
        path = Path(path)
        calibration = {
            'device_type': self.device_type,
            'parameters': self.calibrated_parameters,
            'quality': self.calibration_quality,
            'version': '1.0'
        }
        
        with open(path, 'w') as f:
            json.dump(calibration, f, indent=2)
        
        logger.info(f"Calibration saved to {path}")
    
    def load_calibration(self, path: Union[str, Path]):
        """Load calibrated parameters."""
        path = Path(path)
        
        with open(path, 'r') as f:
            calibration = json.load(f)
        
        self.calibrated_parameters = calibration['parameters']
        self.calibration_quality = calibration.get('quality', {})
        
        logger.info(f"Calibration loaded from {path}")


class PCMCalibrator(DeviceCalibrator):
    """Calibrator for Phase-Change Memory devices."""
    
    def __init__(self):
        super().__init__('pcm_mushroom')
        
        # PCM model: R(T) = R_reset * exp(-E_a * (1/T - 1/T_reset) / k_B)
        self.model_params = {
            'R_set': None,      # Set resistance (Ω)
            'R_reset': None,    # Reset resistance (Ω) 
            'T_set': None,      # Set temperature (K)
            'T_reset': None,    # Reset temperature (K)
            'E_a': None,        # Activation energy (eV)
            'tau_set': None,    # Set time constant (s)
            'tau_reset': None   # Reset time constant (s)
        }
    
    def calibrate(self, data: CalibrationData) -> Dict[str, Any]:
        """Calibrate PCM parameters from I-V and retention data."""
        logger.info(f"Calibrating PCM device: {data.material}")
        
        voltage = data.measurements['voltage']
        current = data.measurements['current']
        resistance = data.measurements['resistance']
        temperature = data.measurements.get('temperature', np.full_like(voltage, 300))
        
        # Extract set and reset states
        set_indices = np.where(resistance < np.median(resistance))[0]
        reset_indices = np.where(resistance > np.median(resistance))[0]
        
        if len(set_indices) == 0 or len(reset_indices) == 0:
            raise ValueError("Could not identify set and reset states")
        
        # Fit basic parameters
        self.model_params['R_set'] = float(np.mean(resistance[set_indices]))
        self.model_params['R_reset'] = float(np.mean(resistance[reset_indices]))
        
        # Fit switching voltages
        V_set = self._find_switching_voltage(voltage, resistance, 'set')
        V_reset = self._find_switching_voltage(voltage, resistance, 'reset')
        
        # Convert to switching temperatures (rough approximation)
        # T_switch ≈ T_ambient + α * V_switch^2 / (thermal_conductance * volume)
        alpha = 1e-3  # Empirical constant
        self.model_params['T_set'] = float(300 + alpha * V_set**2)
        self.model_params['T_reset'] = float(300 + alpha * V_reset**2)
        
        # Fit activation energy from temperature-dependent measurements
        if 'time' in data.measurements:
            self.model_params['E_a'] = self._fit_activation_energy(
                data.measurements['time'],
                resistance,
                temperature
            )
        else:
            # Use typical value for GST225
            self.model_params['E_a'] = 0.6  # eV
        
        # Fit switching time constants
        self.model_params['tau_set'] = 1e-7   # 100 ns typical
        self.model_params['tau_reset'] = 1e-8  # 10 ns typical
        
        self.calibrated_parameters = self.model_params.copy()
        
        # Validate calibration
        self.calibration_quality = self.validate_calibration(data, self.model_params)
        
        logger.info(f"PCM calibration completed. R2 score: {self.calibration_quality.get('r2', 'N/A'):.3f}")
        
        return self.calibrated_parameters
    
    def _find_switching_voltage(self, voltage: np.ndarray, resistance: np.ndarray, switch_type: str) -> float:
        """Find switching voltage from I-V curve."""
        # Calculate resistance derivative
        dR_dV = np.gradient(resistance, voltage)
        
        if switch_type == 'set':
            # Set switching: large negative dR/dV
            switch_idx = np.argmin(dR_dV)
        else:  # reset
            # Reset switching: large positive dR/dV
            switch_idx = np.argmax(dR_dV)
        
        return voltage[switch_idx]
    
    def _fit_activation_energy(self, time: np.ndarray, resistance: np.ndarray, temperature: np.ndarray) -> float:
        """Fit activation energy from retention measurements."""
        # Arrhenius model: R(t) = R0 * exp(-(t/tau) * exp(-E_a/(k_B*T)))
        
        def arrhenius_model(t, R0, tau0, E_a):
            k_B = 8.617e-5  # eV/K
            tau_eff = tau0 * np.exp(E_a / (k_B * temperature))
            return R0 * np.exp(-t / tau_eff)
        
        try:
            # Fit only to retention part (resistance increasing over time)
            retention_mask = resistance > np.mean(resistance)
            if np.sum(retention_mask) < 10:
                logger.warning("Insufficient retention data for E_a fitting")
                return 0.6  # Default value
            
            popt, _ = opt.curve_fit(
                arrhenius_model,
                time[retention_mask],
                resistance[retention_mask],
                p0=[resistance[0], 1e6, 0.6],
                bounds=([0, 1e3, 0.1], [np.inf, 1e9, 2.0])
            )
            
            return float(popt[2])  # E_a
            
        except Exception as e:
            logger.warning(f"Failed to fit activation energy: {e}")
            return 0.6  # Default value for GST225
    
    def validate_calibration(self, data: CalibrationData, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Validate PCM calibration quality."""
        voltage = data.measurements['voltage']
        resistance = data.measurements['resistance']
        
        # Predict resistance using calibrated model
        predicted_resistance = self._predict_resistance(voltage, parameters)
        
        # Calculate metrics
        mse = mean_squared_error(resistance, predicted_resistance)
        r2 = r2_score(resistance, predicted_resistance)
        
        # Calculate relative error
        rel_error = np.mean(np.abs(resistance - predicted_resistance) / resistance)
        
        return {
            'mse': float(mse),
            'r2': float(r2),
            'relative_error': float(rel_error),
            'max_error': float(np.max(np.abs(resistance - predicted_resistance)))
        }
    
    def _predict_resistance(self, voltage: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Predict resistance using calibrated PCM model."""
        R_set = parameters['R_set']
        R_reset = parameters['R_reset']
        T_set = parameters['T_set'] 
        T_reset = parameters['T_reset']
        
        # Simple threshold model for demonstration
        # In practice, would use full thermal-electrical simulation
        
        resistance = np.full_like(voltage, R_reset)
        
        # Set switching
        set_mask = voltage > (T_set - 300) * 0.1  # Rough V-T conversion
        resistance[set_mask] = R_set
        
        # Reset switching
        reset_mask = voltage > (T_reset - 300) * 0.1
        resistance[reset_mask] = R_reset
        
        return resistance


class RRAMCalibrator(DeviceCalibrator):
    """Calibrator for Resistive RAM devices."""
    
    def __init__(self):
        super().__init__('rram_hfo2')
        
        # RRAM model parameters
        self.model_params = {
            'R_on': None,       # ON resistance (Ω)
            'R_off': None,      # OFF resistance (Ω)
            'V_set': None,      # Set voltage (V)
            'V_reset': None,    # Reset voltage (V)
            'beta': None,       # Nonlinearity parameter
            'tau_form': None,   # Forming time constant (s)
            'gamma': None       # Retention parameter
        }
    
    def calibrate(self, data: CalibrationData) -> Dict[str, Any]:
        """Calibrate RRAM parameters from switching characteristics."""
        logger.info(f"Calibrating RRAM device: {data.material}")
        
        voltage = data.measurements['voltage']
        current = data.measurements['current']
        
        # Calculate resistance
        resistance = voltage / (current + 1e-12)  # Avoid division by zero
        
        # Find switching points
        self.model_params['V_set'] = self._find_switching_voltage(voltage, current, 'set')
        self.model_params['V_reset'] = self._find_switching_voltage(voltage, current, 'reset')
        
        # Extract ON/OFF states
        on_indices = np.where(current > np.median(current))[0]
        off_indices = np.where(current < np.median(current))[0]
        
        if len(on_indices) > 0:
            self.model_params['R_on'] = float(np.mean(resistance[on_indices]))
        else:
            self.model_params['R_on'] = 1e3  # Default
        
        if len(off_indices) > 0:
            self.model_params['R_off'] = float(np.mean(resistance[off_indices]))
        else:
            self.model_params['R_off'] = 1e6  # Default
        
        # Fit nonlinearity parameter
        self.model_params['beta'] = self._fit_nonlinearity(voltage, current)
        
        # Default values for temporal parameters
        self.model_params['tau_form'] = 1e-6   # 1 μs
        self.model_params['gamma'] = 0.1       # Retention decay
        
        self.calibrated_parameters = self.model_params.copy()
        
        # Validate calibration
        self.calibration_quality = self.validate_calibration(data, self.model_params)
        
        logger.info(f"RRAM calibration completed. R2 score: {self.calibration_quality.get('r2', 'N/A'):.3f}")
        
        return self.calibrated_parameters
    
    def _find_switching_voltage(self, voltage: np.ndarray, current: np.ndarray, switch_type: str) -> float:
        """Find switching voltage from I-V characteristics."""
        # Calculate current derivative
        dI_dV = np.gradient(current, voltage)
        
        if switch_type == 'set':
            # Set switching: maximum dI/dV
            switch_idx = np.argmax(dI_dV)
        else:  # reset
            # Reset switching: minimum dI/dV (negative)
            switch_idx = np.argmin(dI_dV)
        
        return voltage[switch_idx]
    
    def _fit_nonlinearity(self, voltage: np.ndarray, current: np.ndarray) -> float:
        """Fit nonlinearity parameter β from I-V curve."""
        # RRAM I-V model: I = I0 * sinh(β * V / V_th)
        
        try:
            # Use only positive voltage regime for fitting
            pos_mask = voltage > 0
            if np.sum(pos_mask) < 5:
                logger.warning("Insufficient positive voltage data")
                return 1.0
            
            v_pos = voltage[pos_mask]
            i_pos = current[pos_mask]
            
            # Linear fit in log space: ln(I) = ln(I0) + β * V
            # Approximate sinh(x) ≈ exp(x)/2 for large x
            log_current = np.log(np.abs(i_pos) + 1e-12)
            
            # Linear regression
            A = np.vstack([v_pos, np.ones(len(v_pos))]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, log_current, rcond=None)
            
            return float(np.abs(coeffs[0]))  # β parameter
            
        except Exception as e:
            logger.warning(f"Failed to fit nonlinearity parameter: {e}")
            return 1.0  # Default value
    
    def validate_calibration(self, data: CalibrationData, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Validate RRAM calibration quality."""
        voltage = data.measurements['voltage']
        current = data.measurements['current']
        
        # Predict current using calibrated model
        predicted_current = self._predict_current(voltage, parameters)
        
        # Calculate metrics
        mse = mean_squared_error(current, predicted_current)
        r2 = r2_score(current, predicted_current)
        
        # Calculate relative error (handle near-zero currents)
        nonzero_mask = np.abs(current) > 1e-9
        if np.sum(nonzero_mask) > 0:
            rel_error = np.mean(np.abs(current[nonzero_mask] - predicted_current[nonzero_mask]) / 
                              np.abs(current[nonzero_mask]))
        else:
            rel_error = np.inf
        
        return {
            'mse': float(mse),
            'r2': float(r2),
            'relative_error': float(rel_error),
            'max_error': float(np.max(np.abs(current - predicted_current)))
        }
    
    def _predict_current(self, voltage: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Predict current using calibrated RRAM model."""
        R_on = parameters['R_on']
        R_off = parameters['R_off']
        V_set = parameters['V_set']
        V_reset = parameters['V_reset']
        beta = parameters['beta']
        
        # Simplified switching model
        current = np.zeros_like(voltage)
        
        for i, v in enumerate(voltage):
            if v > V_set:
                # ON state
                R_eff = R_on
            elif v < V_reset:
                # OFF state  
                R_eff = R_off
            else:
                # Transition region - linear interpolation
                alpha = (v - V_reset) / (V_set - V_reset)
                R_eff = R_off * (1 - alpha) + R_on * alpha
            
            # Apply nonlinearity
            if R_eff > 0:
                current[i] = v / R_eff * np.tanh(beta * v)
        
        return current


class PhotonicCalibrator(DeviceCalibrator):
    """Calibrator for photonic phase shifters."""
    
    def __init__(self, shifter_type: str):
        super().__init__(f'{shifter_type}_phase_shifter')
        self.shifter_type = shifter_type
        
        if shifter_type == 'thermal':
            self.model_params = {
                'power_per_pi': None,     # Power for π phase shift (W)
                'response_time': None,    # Thermal time constant (s)
                'thermal_crosstalk': None # Crosstalk coefficient
            }
        elif shifter_type == 'plasma':
            self.model_params = {
                'voltage_per_pi': None,   # Voltage for π phase shift (V)
                'capacitance': None,      # Device capacitance (F)
                'bandwidth': None         # Electrical bandwidth (Hz)
            }
    
    def calibrate(self, data: CalibrationData) -> Dict[str, Any]:
        """Calibrate photonic phase shifter parameters."""
        logger.info(f"Calibrating {self.shifter_type} phase shifter")
        
        if self.shifter_type == 'thermal':
            return self._calibrate_thermal(data)
        elif self.shifter_type == 'plasma':
            return self._calibrate_plasma(data)
        else:
            raise ValueError(f"Unknown shifter type: {self.shifter_type}")
    
    def _calibrate_thermal(self, data: CalibrationData) -> Dict[str, Any]:
        """Calibrate thermal phase shifter."""
        power = data.measurements['power']
        phase_shift = data.measurements['phase_shift']
        
        # Fit linear relationship: φ = η * P
        # Find power for π phase shift
        pi_indices = np.where(np.abs(phase_shift - np.pi) < 0.1)[0]
        if len(pi_indices) > 0:
            self.model_params['power_per_pi'] = float(np.mean(power[pi_indices]))
        else:
            # Linear fit
            coeffs = np.polyfit(power, phase_shift, 1)
            self.model_params['power_per_pi'] = float(np.pi / coeffs[0])
        
        # Fit response time from temporal data
        if 'response_time' in data.measurements:
            self.model_params['response_time'] = float(np.mean(data.measurements['response_time']))
        else:
            self.model_params['response_time'] = 10e-6  # Default 10 μs
        
        # Estimate thermal crosstalk
        self.model_params['thermal_crosstalk'] = 0.05  # Default 5%
        
        self.calibrated_parameters = self.model_params.copy()
        self.calibration_quality = self.validate_calibration(data, self.model_params)
        
        return self.calibrated_parameters
    
    def _calibrate_plasma(self, data: CalibrationData) -> Dict[str, Any]:
        """Calibrate plasma dispersion phase shifter."""
        voltage = data.measurements['voltage']
        phase_shift = data.measurements['phase_shift']
        
        # Fit quadratic relationship: φ = α * V^2
        coeffs = np.polyfit(voltage**2, phase_shift, 1)
        
        # Find voltage for π phase shift
        self.model_params['voltage_per_pi'] = float(np.sqrt(np.pi / coeffs[0]))
        
        # Estimate capacitance from device geometry (if available)
        if 'capacitance' in data.measurements:
            self.model_params['capacitance'] = float(np.mean(data.measurements['capacitance']))
        else:
            self.model_params['capacitance'] = 10e-15  # Default 10 fF
        
        # Estimate bandwidth
        self.model_params['bandwidth'] = 1e9  # Default 1 GHz
        
        self.calibrated_parameters = self.model_params.copy()
        self.calibration_quality = self.validate_calibration(data, self.model_params)
        
        return self.calibrated_parameters
    
    def validate_calibration(self, data: CalibrationData, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Validate photonic calibration quality."""
        if self.shifter_type == 'thermal':
            power = data.measurements['power']
            phase_shift = data.measurements['phase_shift']
            
            # Predict phase shift
            predicted_phase = power * np.pi / parameters['power_per_pi']
            
            return {
                'mse': float(mean_squared_error(phase_shift, predicted_phase)),
                'r2': float(r2_score(phase_shift, predicted_phase)),
                'linearity_error': float(np.std(phase_shift - predicted_phase))
            }
        
        elif self.shifter_type == 'plasma':
            voltage = data.measurements['voltage']
            phase_shift = data.measurements['phase_shift']
            
            # Predict phase shift (quadratic model)
            V_pi = parameters['voltage_per_pi']
            predicted_phase = np.pi * (voltage / V_pi)**2
            
            return {
                'mse': float(mean_squared_error(phase_shift, predicted_phase)),
                'r2': float(r2_score(phase_shift, predicted_phase)),
                'quadratic_error': float(np.std(phase_shift - predicted_phase))
            }
        
        return {}


class CalibrationManager:
    """Manages multiple device calibrations."""
    
    def __init__(self, calibration_dir: Union[str, Path]):
        self.calibration_dir = Path(calibration_dir)
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        
        self.calibrators = {
            'pcm_mushroom': PCMCalibrator(),
            'rram_hfo2': RRAMCalibrator(),
            'thermal_phase_shifter': PhotonicCalibrator('thermal'),
            'plasma_phase_shifter': PhotonicCalibrator('plasma')
        }
        
        self.calibrated_devices = {}
    
    def add_calibration_data(
        self,
        device_id: str,
        data_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Add and process calibration data for a device."""
        data = CalibrationData.load(data_path)
        
        if data.device_type not in self.calibrators:
            raise ValueError(f"No calibrator available for {data.device_type}")
        
        calibrator = self.calibrators[data.device_type]
        parameters = calibrator.calibrate(data)
        
        # Save calibration
        calib_path = self.calibration_dir / f"{device_id}_calibration.json"
        calibrator.save_calibration(calib_path)
        
        self.calibrated_devices[device_id] = {
            'device_type': data.device_type,
            'parameters': parameters,
            'quality': calibrator.calibration_quality,
            'calibration_path': str(calib_path)
        }
        
        logger.info(f"Calibrated device {device_id}")
        
        return self.calibrated_devices[device_id]
    
    def get_device_parameters(self, device_id: str) -> Dict[str, Any]:
        """Get calibrated parameters for a device."""
        if device_id not in self.calibrated_devices:
            # Try to load from file
            calib_path = self.calibration_dir / f"{device_id}_calibration.json"
            if calib_path.exists():
                with open(calib_path, 'r') as f:
                    calib_data = json.load(f)
                
                self.calibrated_devices[device_id] = {
                    'device_type': calib_data['device_type'],
                    'parameters': calib_data['parameters'],
                    'quality': calib_data.get('quality', {}),
                    'calibration_path': str(calib_path)
                }
            else:
                raise ValueError(f"No calibration found for device {device_id}")
        
        return self.calibrated_devices[device_id]['parameters']
    
    def validate_all_calibrations(self) -> Dict[str, Dict[str, float]]:
        """Validate quality of all calibrations."""
        validation_results = {}
        
        for device_id, device_info in self.calibrated_devices.items():
            quality = device_info.get('quality', {})
            validation_results[device_id] = quality
        
        return validation_results
    
    def generate_calibration_report(self, output_path: Union[str, Path]):
        """Generate comprehensive calibration report."""
        report = {
            'calibrated_devices': len(self.calibrated_devices),
            'device_summary': {},
            'quality_metrics': self.validate_all_calibrations(),
            'recommendations': []
        }
        
        # Summarize by device type
        for device_id, device_info in self.calibrated_devices.items():
            device_type = device_info['device_type']
            if device_type not in report['device_summary']:
                report['device_summary'][device_type] = {'count': 0, 'avg_quality': {}}
            
            report['device_summary'][device_type]['count'] += 1
        
        # Add recommendations based on calibration quality
        for device_id, quality in report['quality_metrics'].items():
            r2 = quality.get('r2', 0)
            if r2 < 0.9:
                report['recommendations'].append(
                    f"Device {device_id}: Low R² ({r2:.3f}) - consider recalibration with more data"
                )
            
            rel_error = quality.get('relative_error', np.inf)
            if rel_error > 0.1:
                report['recommendations'].append(
                    f"Device {device_id}: High relative error ({rel_error:.1%}) - check measurement accuracy"
                )
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Calibration report saved to {output_path}")
        return report