"""
Enhanced monitoring and health check system for PhoMem-CoSim.
"""

import time
import threading
import psutil
import traceback
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import queue
import warnings

try:
    import jax.numpy as jnp
    import numpy as np
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = None
    np = None

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_usage: float
    network_io: Dict[str, int] = field(default_factory=dict)
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None

@dataclass
class SimulationMetrics:
    """Simulation-specific metrics."""
    timestamp: float
    iteration: int
    convergence_error: float
    computation_time: float
    memory_peak: float
    network_accuracy: Optional[float] = None
    power_consumption: Optional[float] = None
    optical_losses: Optional[float] = None

@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: str  # 'healthy', 'warning', 'critical'
    message: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)

class PerformanceMonitor:
    """Real-time performance monitoring for PhoMem-CoSim."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.monitoring_active = False
        self.monitor_thread = None
        self.metrics_queue = queue.Queue(maxsize=1000)
        self.system_metrics_history = []
        self.simulation_metrics_history = []
        self.max_history_length = 1000
        
        # Monitoring callbacks
        self.callbacks = []
        
        # Thresholds
        self.thresholds = {
            'cpu_usage': 90.0,
            'memory_usage': 90.0,
            'disk_usage': 95.0,
            'convergence_error': 1e-3,
            'max_computation_time': 3600.0  # 1 hour
        }
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        print("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.system_metrics_history.append(metrics)
                
                # Trim history
                if len(self.system_metrics_history) > self.max_history_length:
                    self.system_metrics_history.pop(0)
                
                # Check thresholds
                self._check_thresholds(metrics)
                
                # Run callbacks
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        print(f"Monitoring callback error: {e}")
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU and memory
            cpu_usage = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv
            } if network else {}
            
            # GPU metrics (if available)
            gpu_usage = None
            gpu_memory = None
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_usage = gpu_info.gpu
                gpu_memory = gpu_mem_info.used / gpu_mem_info.total * 100
            except ImportError:
                pass
            except Exception:
                pass
            
            return SystemMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                memory_available=memory.available,
                disk_usage=disk.percent,
                network_io=network_io,
                gpu_usage=gpu_usage,
                gpu_memory=gpu_memory
            )
            
        except Exception as e:
            print(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                timestamp=time.time(),
                cpu_usage=0.0,
                memory_usage=0.0,
                memory_available=0,
                disk_usage=0.0
            )
    
    def log_simulation_metrics(self, 
                             iteration: int,
                             convergence_error: float,
                             computation_time: float,
                             network_accuracy: Optional[float] = None,
                             power_consumption: Optional[float] = None,
                             optical_losses: Optional[float] = None):
        """Log simulation-specific metrics."""
        try:
            # Get current memory usage
            process = psutil.Process()
            memory_peak = process.memory_info().rss / 1024 / 1024  # MB
            
            metrics = SimulationMetrics(
                timestamp=time.time(),
                iteration=iteration,
                convergence_error=convergence_error,
                computation_time=computation_time,
                memory_peak=memory_peak,
                network_accuracy=network_accuracy,
                power_consumption=power_consumption,
                optical_losses=optical_losses
            )
            
            self.simulation_metrics_history.append(metrics)
            
            # Trim history
            if len(self.simulation_metrics_history) > self.max_history_length:
                self.simulation_metrics_history.pop(0)
            
            # Check simulation thresholds
            self._check_simulation_thresholds(metrics)
            
        except Exception as e:
            print(f"Failed to log simulation metrics: {e}")
    
    def _check_thresholds(self, metrics: SystemMetrics):
        """Check system metrics against thresholds."""
        if metrics.cpu_usage > self.thresholds['cpu_usage']:
            self._trigger_alert(
                'system',
                f"High CPU usage: {metrics.cpu_usage:.1f}%",
                'warning'
            )
        
        if metrics.memory_usage > self.thresholds['memory_usage']:
            self._trigger_alert(
                'system',
                f"High memory usage: {metrics.memory_usage:.1f}%",
                'warning'
            )
        
        if metrics.disk_usage > self.thresholds['disk_usage']:
            self._trigger_alert(
                'system',
                f"High disk usage: {metrics.disk_usage:.1f}%",
                'critical'
            )
    
    def _check_simulation_thresholds(self, metrics: SimulationMetrics):
        """Check simulation metrics against thresholds."""
        if metrics.convergence_error > self.thresholds['convergence_error']:
            self._trigger_alert(
                'simulation',
                f"Poor convergence: error = {metrics.convergence_error:.2e}",
                'warning'
            )
        
        if metrics.computation_time > self.thresholds['max_computation_time']:
            self._trigger_alert(
                'simulation',
                f"Long computation time: {metrics.computation_time:.1f}s",
                'warning'
            )
    
    def _trigger_alert(self, category: str, message: str, level: str):
        """Trigger monitoring alert."""
        alert = {
            'timestamp': time.time(),
            'category': category,
            'message': message,
            'level': level
        }
        
        # For now, just print alerts
        print(f"ALERT [{level.upper()}] {category}: {message}")
        
        # Could extend to send notifications, log to file, etc.
    
    def add_callback(self, callback: Callable[[SystemMetrics], None]):
        """Add monitoring callback function."""
        self.callbacks.append(callback)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of system performance."""
        if not self.system_metrics_history:
            return {'status': 'no_data'}
        
        recent_metrics = self.system_metrics_history[-10:]  # Last 10 measurements
        
        return {
            'status': 'active',
            'current_cpu': recent_metrics[-1].cpu_usage,
            'avg_cpu': sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            'current_memory': recent_metrics[-1].memory_usage,
            'avg_memory': sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            'disk_usage': recent_metrics[-1].disk_usage,
            'gpu_usage': recent_metrics[-1].gpu_usage,
            'measurements_count': len(self.system_metrics_history),
            'monitoring_duration': time.time() - self.system_metrics_history[0].timestamp if self.system_metrics_history else 0
        }
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get summary of simulation performance."""
        if not self.simulation_metrics_history:
            return {'status': 'no_simulations'}
        
        metrics = self.simulation_metrics_history
        
        return {
            'status': 'active',
            'total_iterations': len(metrics),
            'avg_computation_time': sum(m.computation_time for m in metrics) / len(metrics),
            'final_convergence_error': metrics[-1].convergence_error,
            'peak_memory_usage': max(m.memory_peak for m in metrics),
            'best_accuracy': max((m.network_accuracy for m in metrics if m.network_accuracy is not None), default=None),
            'avg_power_consumption': sum((m.power_consumption for m in metrics if m.power_consumption is not None)) / len([m for m in metrics if m.power_consumption is not None]) if any(m.power_consumption is not None for m in metrics) else None
        }

class HealthChecker:
    """System health checking for PhoMem-CoSim."""
    
    def __init__(self):
        self.health_checks = {}
        self.last_check_results = {}
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check('dependencies', self._check_dependencies)
        self.register_check('memory', self._check_memory)
        self.register_check('disk_space', self._check_disk_space)
        self.register_check('jax_functionality', self._check_jax)
    
    def register_check(self, name: str, check_function: Callable[[], HealthCheck]):
        """Register a health check function."""
        self.health_checks[name] = check_function
    
    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[name] = result
                self.last_check_results[name] = result
            except Exception as e:
                error_result = HealthCheck(
                    name=name,
                    status='critical',
                    message=f"Health check failed: {str(e)}",
                    timestamp=time.time(),
                    context={'error': str(e), 'traceback': traceback.format_exc()}
                )
                results[name] = error_result
                self.last_check_results[name] = error_result
        
        return results
    
    def run_check(self, name: str) -> HealthCheck:
        """Run a specific health check."""
        if name not in self.health_checks:
            return HealthCheck(
                name=name,
                status='critical',
                message=f"Unknown health check: {name}",
                timestamp=time.time()
            )
        
        try:
            result = self.health_checks[name]()
            self.last_check_results[name] = result
            return result
        except Exception as e:
            error_result = HealthCheck(
                name=name,
                status='critical',
                message=f"Health check failed: {str(e)}",
                timestamp=time.time(),
                context={'error': str(e)}
            )
            self.last_check_results[name] = error_result
            return error_result
    
    def _check_dependencies(self) -> HealthCheck:
        """Check critical dependencies."""
        missing_deps = []
        
        try:
            import jax
            import jaxlib
        except ImportError:
            missing_deps.append('JAX')
        
        try:
            import numpy
        except ImportError:
            missing_deps.append('NumPy')
        
        try:
            import matplotlib
        except ImportError:
            missing_deps.append('matplotlib')
        
        if missing_deps:
            return HealthCheck(
                name='dependencies',
                status='critical',
                message=f"Missing dependencies: {', '.join(missing_deps)}",
                timestamp=time.time(),
                context={'missing_dependencies': missing_deps}
            )
        
        return HealthCheck(
            name='dependencies',
            status='healthy',
            message="All critical dependencies available",
            timestamp=time.time()
        )
    
    def _check_memory(self) -> HealthCheck:
        """Check available memory."""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / 1024**3
            
            if available_gb < 0.5:  # Less than 500MB
                return HealthCheck(
                    name='memory',
                    status='critical',
                    message=f"Very low memory: {available_gb:.1f} GB available",
                    timestamp=time.time(),
                    context={'available_gb': available_gb}
                )
            elif available_gb < 2.0:  # Less than 2GB
                return HealthCheck(
                    name='memory',
                    status='warning',
                    message=f"Low memory: {available_gb:.1f} GB available",
                    timestamp=time.time(),
                    context={'available_gb': available_gb}
                )
            else:
                return HealthCheck(
                    name='memory',
                    status='healthy',
                    message=f"Memory OK: {available_gb:.1f} GB available",
                    timestamp=time.time(),
                    context={'available_gb': available_gb}
                )
        
        except Exception as e:
            return HealthCheck(
                name='memory',
                status='critical',
                message=f"Failed to check memory: {str(e)}",
                timestamp=time.time()
            )
    
    def _check_disk_space(self) -> HealthCheck:
        """Check available disk space."""
        try:
            disk = psutil.disk_usage('/')
            available_gb = disk.free / 1024**3
            usage_percent = (disk.used / disk.total) * 100
            
            if usage_percent > 95:
                return HealthCheck(
                    name='disk_space',
                    status='critical',
                    message=f"Disk full: {usage_percent:.1f}% used, {available_gb:.1f} GB free",
                    timestamp=time.time(),
                    context={'usage_percent': usage_percent, 'available_gb': available_gb}
                )
            elif usage_percent > 90:
                return HealthCheck(
                    name='disk_space',
                    status='warning',
                    message=f"Disk almost full: {usage_percent:.1f}% used, {available_gb:.1f} GB free",
                    timestamp=time.time(),
                    context={'usage_percent': usage_percent, 'available_gb': available_gb}
                )
            else:
                return HealthCheck(
                    name='disk_space',
                    status='healthy',
                    message=f"Disk OK: {usage_percent:.1f}% used, {available_gb:.1f} GB free",
                    timestamp=time.time(),
                    context={'usage_percent': usage_percent, 'available_gb': available_gb}
                )
        
        except Exception as e:
            return HealthCheck(
                name='disk_space',
                status='critical',
                message=f"Failed to check disk space: {str(e)}",
                timestamp=time.time()
            )
    
    def _check_jax(self) -> HealthCheck:
        """Check JAX functionality."""
        if not JAX_AVAILABLE:
            return HealthCheck(
                name='jax_functionality',
                status='critical',
                message="JAX not available",
                timestamp=time.time()
            )
        
        try:
            # Test basic JAX operations
            x = jnp.array([1.0, 2.0, 3.0])
            y = jnp.sum(x)
            
            # Test GPU/TPU availability
            devices = jax.devices()
            device_types = [str(d).split(':')[0] for d in devices]
            
            return HealthCheck(
                name='jax_functionality',
                status='healthy',
                message=f"JAX working, devices: {', '.join(device_types)}",
                timestamp=time.time(),
                context={'devices': device_types, 'test_result': float(y)}
            )
        
        except Exception as e:
            return HealthCheck(
                name='jax_functionality',
                status='warning',
                message=f"JAX issues: {str(e)}",
                timestamp=time.time(),
                context={'error': str(e)}
            )
    
    def get_overall_health(self) -> str:
        """Get overall system health status."""
        if not self.last_check_results:
            return 'unknown'
        
        statuses = [check.status for check in self.last_check_results.values()]
        
        if 'critical' in statuses:
            return 'critical'
        elif 'warning' in statuses:
            return 'warning'
        else:
            return 'healthy'

# Global instances
_performance_monitor = PerformanceMonitor()
_health_checker = HealthChecker()

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    return _performance_monitor

def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    return _health_checker

def start_monitoring():
    """Start system monitoring."""
    _performance_monitor.start_monitoring()

def stop_monitoring():
    """Stop system monitoring."""
    _performance_monitor.stop_monitoring()

def run_health_check() -> Dict[str, HealthCheck]:
    """Run all health checks."""
    return _health_checker.run_all_checks()

def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status."""
    health_results = _health_checker.run_all_checks()
    
    return {
        'timestamp': time.time(),
        'overall_health': _health_checker.get_overall_health(),
        'health_checks': {name: {
            'status': check.status,
            'message': check.message,
            'timestamp': check.timestamp
        } for name, check in health_results.items()},
        'performance': _performance_monitor.get_system_summary(),
        'simulation': _performance_monitor.get_simulation_summary()
    }