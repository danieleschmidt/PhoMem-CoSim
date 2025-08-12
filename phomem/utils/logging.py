"""
Comprehensive logging system for PhoMem-CoSim.
"""

import logging
import sys
import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union
from pathlib import Path
import threading
from contextlib import contextmanager


class PhoMemLogger:
    """Centralized logging system for PhoMem-CoSim."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.loggers = {}
        self.log_dir = Path.home() / '.phomem' / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.performance_log = {}
        self.start_times = {}
        
        # Setup main logger
        self._setup_main_logger()
    
    def _setup_main_logger(self):
        """Setup the main PhoMem logger."""
        logger = logging.getLogger('phomem')
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f'phomem_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Error handler (separate file for errors)
        error_file = self.log_dir / f'phomem_errors_{datetime.now().strftime("%Y%m%d")}.log'
        error_handler = logging.FileHandler(error_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
        
        self.loggers['main'] = logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger for a specific module."""
        if name not in self.loggers:
            logger = logging.getLogger(f'phomem.{name}')
            logger.setLevel(logging.DEBUG)
            
            # Inherit handlers from main logger
            main_logger = self.loggers['main']
            for handler in main_logger.handlers:
                logger.addHandler(handler)
            
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    @contextmanager
    def performance_timer(self, operation: str, metadata: Dict[str, Any] = None):
        """Context manager for performance timing."""
        metadata = metadata or {}
        start_time = time.perf_counter()
        logger = self.get_logger('performance')
        
        try:
            logger.info(f"Starting {operation}", extra={'metadata': metadata})
            yield
        except Exception as e:
            logger.error(f"Error in {operation}: {e}", extra={'metadata': metadata})
            raise
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Log performance
            perf_data = {
                'operation': operation,
                'duration': duration,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata
            }
            
            logger.info(f"Completed {operation} in {duration:.4f}s", extra=perf_data)
            
            # Store performance data
            if operation not in self.performance_log:
                self.performance_log[operation] = []
            self.performance_log[operation].append(perf_data)
    
    def log_simulation_start(self, sim_type: str, parameters: Dict[str, Any]):
        """Log simulation start."""
        logger = self.get_logger('simulation')
        logger.info(f"Starting {sim_type} simulation", extra={
            'simulation_type': sim_type,
            'parameters': parameters,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    def log_simulation_end(self, sim_type: str, results: Dict[str, Any]):
        """Log simulation completion."""
        logger = self.get_logger('simulation')
        logger.info(f"Completed {sim_type} simulation", extra={
            'simulation_type': sim_type,
            'results': results,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    def log_device_operation(self, device_type: str, operation: str, 
                           params: Dict[str, Any], result: Any = None):
        """Log device operations for debugging."""
        logger = self.get_logger('devices')
        logger.debug(f"{device_type} {operation}", extra={
            'device_type': device_type,
            'operation': operation,
            'parameters': params,
            'result': str(result) if result is not None else None
        })
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log errors with context."""
        logger = self.get_logger('errors')
        context = context or {}
        
        logger.error(f"Error occurred: {error}", extra={
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }, exc_info=True)
    
    def log_warning(self, message: str, context: Dict[str, Any] = None):
        """Log warnings with context."""
        logger = self.get_logger('warnings')
        logger.warning(message, extra={
            'context': context or {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        summary = {}
        
        for operation, records in self.performance_log.items():
            durations = [r['duration'] for r in records]
            summary[operation] = {
                'count': len(durations),
                'total_time': sum(durations),
                'avg_time': sum(durations) / len(durations),
                'min_time': min(durations),
                'max_time': max(durations)
            }
        
        return summary
    
    def save_performance_report(self, filepath: Optional[Union[str, Path]] = None):
        """Save performance report to file."""
        if filepath is None:
            filepath = self.log_dir / f'performance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'summary': self.get_performance_summary(),
            'detailed_log': self.performance_log
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger = self.get_logger('reports')
        logger.info(f"Performance report saved to {filepath}")


class SimulationLogger:
    """Specialized logger for simulation tracking."""
    
    def __init__(self, simulation_id: str):
        self.simulation_id = simulation_id
        self.logger = PhoMemLogger().get_logger(f'sim_{simulation_id}')
        self.start_time = time.time()
        self.checkpoints = []
    
    def checkpoint(self, name: str, data: Dict[str, Any] = None):
        """Log a simulation checkpoint."""
        timestamp = time.time()
        elapsed = timestamp - self.start_time
        
        checkpoint_data = {
            'name': name,
            'timestamp': timestamp,
            'elapsed_time': elapsed,
            'data': data or {}
        }
        
        self.checkpoints.append(checkpoint_data)
        self.logger.info(f"Checkpoint: {name} (t={elapsed:.3f}s)", extra=checkpoint_data)
    
    def log_convergence(self, iteration: int, error: float, target: float):
        """Log convergence information."""
        self.logger.debug(f"Iteration {iteration}: error={error:.2e}, target={target:.2e}")
    
    def log_physics_coupling(self, physics_domains: list, coupling_strength: float):
        """Log multi-physics coupling information."""
        self.logger.info(f"Physics coupling: {physics_domains}, strength={coupling_strength}")
    
    def finalize(self, success: bool, final_results: Dict[str, Any] = None):
        """Finalize simulation logging."""
        total_time = time.time() - self.start_time
        
        self.logger.info(f"Simulation {self.simulation_id} {'completed' if success else 'failed'} "
                        f"in {total_time:.3f}s", extra={
            'simulation_id': self.simulation_id,
            'success': success,
            'total_time': total_time,
            'checkpoints': len(self.checkpoints),
            'final_results': final_results or {}
        })


# Global logger instance
_logger_instance = None

def get_logger(name: str = 'main') -> logging.Logger:
    """Get a PhoMem logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = PhoMemLogger()
    return _logger_instance.get_logger(name)

def performance_timer(operation: str, metadata: Dict[str, Any] = None):
    """Decorator or context manager for performance timing."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = PhoMemLogger()
    return _logger_instance.performance_timer(operation, metadata)

def log_error(error: Exception, context: Dict[str, Any] = None):
    """Convenience function for error logging."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = PhoMemLogger()
    _logger_instance.log_error(error, context)

def log_warning(message: str, context: Dict[str, Any] = None):
    """Convenience function for warning logging."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = PhoMemLogger()
    _logger_instance.log_warning(message, context)

def create_simulation_logger(simulation_id: str) -> SimulationLogger:
    """Create a simulation-specific logger."""
    return SimulationLogger(simulation_id)

def setup_logging(level=logging.INFO):
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Initialize the global logger instance
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = PhoMemLogger()