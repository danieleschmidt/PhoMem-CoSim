"""
Intelligent Monitoring and Observability System - AUTONOMOUS SDLC v4.0
Advanced real-time monitoring with predictive analytics and anomaly detection.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import chex
from functools import partial
import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from pathlib import Path
import psutil
import logging

from .utils.logging import get_logger
from .utils.performance import PerformanceOptimizer


class MetricType(Enum):
    """Types of metrics to monitor."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class MetricPoint:
    """Individual metric measurement."""
    
    timestamp: float
    value: Union[float, int]
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert definition."""
    
    alert_id: str
    severity: AlertSeverity
    message: str
    timestamp: float
    source: str
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[float] = None


class MetricsCollector:
    """High-performance metrics collection system."""
    
    def __init__(self, max_points_per_metric: int = 10000):
        self.metrics = defaultdict(deque)
        self.max_points = max_points_per_metric
        self.logger = get_logger(__name__)
        self._lock = threading.Lock()
        
    def record_metric(self, 
                     metric_name: str,
                     value: Union[float, int],
                     metric_type: MetricType = MetricType.GAUGE,
                     labels: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a metric point."""
        
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {},
            metadata=metadata or {}
        )
        
        with self._lock:
            metric_key = f"{metric_name}:{metric_type.value}"
            self.metrics[metric_key].append(point)
            
            # Maintain maximum points per metric
            if len(self.metrics[metric_key]) > self.max_points:
                self.metrics[metric_key].popleft()
    
    def get_metric_history(self, 
                          metric_name: str,
                          metric_type: MetricType = MetricType.GAUGE,
                          time_window: Optional[float] = None) -> List[MetricPoint]:
        """Get historical metric data."""
        
        metric_key = f"{metric_name}:{metric_type.value}"
        
        with self._lock:
            points = list(self.metrics[metric_key])
        
        if time_window is not None:
            cutoff_time = time.time() - time_window
            points = [p for p in points if p.timestamp >= cutoff_time]
        
        return points
    
    def compute_statistics(self, 
                          metric_name: str,
                          metric_type: MetricType = MetricType.GAUGE,
                          time_window: float = 3600) -> Dict[str, float]:
        """Compute statistical summary of metric."""
        
        points = self.get_metric_history(metric_name, metric_type, time_window)
        
        if not points:
            return {}
        
        values = [p.value for p in points]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }


class AnomalyDetector:
    """Machine learning-based anomaly detection for metrics."""
    
    def __init__(self, 
                 detection_window: int = 100,
                 anomaly_threshold: float = 3.0):
        self.detection_window = detection_window
        self.anomaly_threshold = anomaly_threshold
        self.baseline_models = {}
        self.logger = get_logger(__name__)
        
    def update_baseline(self, 
                       metric_name: str,
                       historical_data: List[float]) -> None:
        """Update baseline model for anomaly detection."""
        
        if len(historical_data) < 10:
            return
        
        # Simple statistical baseline (in production, use more sophisticated ML)
        data_array = np.array(historical_data)
        
        # Remove outliers for baseline calculation
        q75, q25 = np.percentile(data_array, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        clean_data = data_array[(data_array >= lower_bound) & (data_array <= upper_bound)]
        
        if len(clean_data) > 5:
            self.baseline_models[metric_name] = {
                'mean': float(np.mean(clean_data)),
                'std': float(np.std(clean_data)),
                'min': float(np.min(clean_data)),
                'max': float(np.max(clean_data)),
                'trend': self._compute_trend(clean_data)
            }
    
    def detect_anomaly(self, 
                      metric_name: str,
                      current_value: float,
                      recent_values: List[float]) -> Tuple[bool, float, str]:
        """Detect if current value is anomalous."""
        
        if metric_name not in self.baseline_models:
            return False, 0.0, "No baseline model"
        
        baseline = self.baseline_models[metric_name]
        
        # Statistical anomaly detection
        z_score = abs(current_value - baseline['mean']) / max(baseline['std'], 1e-6)
        
        # Range-based detection
        range_violation = (current_value < baseline['min'] * 0.5 or 
                          current_value > baseline['max'] * 2.0)
        
        # Trend-based detection
        if len(recent_values) >= 5:
            recent_trend = self._compute_trend(recent_values)
            trend_change = abs(recent_trend - baseline['trend'])
            significant_trend_change = trend_change > baseline['std']
        else:
            significant_trend_change = False
        
        # Combine detections
        is_anomaly = (z_score > self.anomaly_threshold or 
                     range_violation or 
                     significant_trend_change)
        
        confidence = min(z_score / self.anomaly_threshold, 1.0)
        
        reason = []
        if z_score > self.anomaly_threshold:
            reason.append(f"statistical_outlier(z={z_score:.2f})")
        if range_violation:
            reason.append("range_violation")
        if significant_trend_change:
            reason.append("trend_anomaly")
        
        return is_anomaly, confidence, "|".join(reason)
    
    def _compute_trend(self, values: np.ndarray) -> float:
        """Compute trend slope of values."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        trend = np.polyfit(x, values, 1)[0]
        return float(trend)


class AlertManager:
    """Intelligent alert management with suppression and escalation."""
    
    def __init__(self, 
                 max_alerts: int = 1000,
                 suppression_window: float = 300):  # 5 minutes
        self.active_alerts = {}
        self.alert_history = deque(maxlen=max_alerts)
        self.suppression_window = suppression_window
        self.escalation_rules = {}
        self.logger = get_logger(__name__)
        
    def trigger_alert(self, 
                     alert_id: str,
                     severity: AlertSeverity,
                     message: str,
                     source: str,
                     labels: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Trigger a new alert with intelligent suppression."""
        
        # Check for duplicate suppression
        if self._is_suppressed(alert_id, source):
            self.logger.debug(f"Alert {alert_id} suppressed (too recent)")
            return False
        
        # Create alert
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            message=message,
            timestamp=time.time(),
            source=source,
            labels=labels or {},
            metadata=metadata or {}
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Log alert
        self.logger.warning(f"ALERT [{severity.value}] {alert_id}: {message}")
        
        # Check escalation rules
        self._check_escalation(alert)
        
        return True
    
    def resolve_alert(self, alert_id: str, resolution_note: str = "") -> bool:
        """Mark alert as resolved."""
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_timestamp = time.time()
            alert.metadata['resolution_note'] = resolution_note
            
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Alert {alert_id} resolved: {resolution_note}")
            return True
        
        return False
    
    def get_active_alerts(self, 
                         severity_filter: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get currently active alerts."""
        
        alerts = list(self.active_alerts.values())
        
        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def _is_suppressed(self, alert_id: str, source: str) -> bool:
        """Check if alert should be suppressed."""
        
        current_time = time.time()
        
        # Check recent alerts for same alert_id from same source
        for alert in reversed(self.alert_history):
            if (alert.alert_id == alert_id and 
                alert.source == source and
                current_time - alert.timestamp < self.suppression_window):
                return True
        
        return False
    
    def _check_escalation(self, alert: Alert) -> None:
        """Check if alert should be escalated."""
        
        # Count recent alerts of same type
        recent_count = sum(1 for a in reversed(self.alert_history)
                          if (a.alert_id == alert.alert_id and
                              time.time() - a.timestamp < 3600))  # 1 hour
        
        # Escalate if too many similar alerts
        if recent_count >= 5 and alert.severity != AlertSeverity.EMERGENCY:
            escalated_alert = Alert(
                alert_id=f"{alert.alert_id}_escalated",
                severity=AlertSeverity.EMERGENCY,
                message=f"ESCALATED: {recent_count} occurrences of {alert.alert_id}",
                timestamp=time.time(),
                source=alert.source,
                labels=alert.labels,
                metadata={'original_alert': alert.alert_id, 'occurrence_count': recent_count}
            )
            
            self.active_alerts[escalated_alert.alert_id] = escalated_alert
            self.alert_history.append(escalated_alert)
            
            self.logger.critical(f"Alert escalated: {escalated_alert.message}")


class SystemHealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self, 
                 collection_interval: float = 5.0,
                 enable_gpu_monitoring: bool = True):
        self.collection_interval = collection_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.metrics_collector = MetricsCollector()
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()
        self.logger = get_logger(__name__)
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
    def start_monitoring(self) -> None:
        """Start background health monitoring."""
        
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            self.logger.warning("Monitoring already running")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
        
        self.logger.info("System health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10)
        
        self.logger.info("System health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        
        while not self._stop_monitoring.is_set():
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect JAX/GPU metrics
                if self.enable_gpu_monitoring:
                    self._collect_gpu_metrics()
                
                # Collect simulation-specific metrics
                self._collect_simulation_metrics()
                
                # Run anomaly detection
                self._run_anomaly_detection()
                
                # Sleep until next collection
                self._stop_monitoring.wait(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self._stop_monitoring.wait(self.collection_interval)
    
    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics_collector.record_metric("system.cpu.usage_percent", cpu_percent)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics_collector.record_metric("system.memory.usage_percent", memory.percent)
        self.metrics_collector.record_metric("system.memory.available_gb", memory.available / 1024**3)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.metrics_collector.record_metric("system.disk.usage_percent", 
                                           (disk.used / disk.total) * 100)
        
        # Network metrics
        net_io = psutil.net_io_counters()
        self.metrics_collector.record_metric("system.network.bytes_sent", 
                                           net_io.bytes_sent, MetricType.COUNTER)
        self.metrics_collector.record_metric("system.network.bytes_recv", 
                                           net_io.bytes_recv, MetricType.COUNTER)
    
    def _collect_gpu_metrics(self) -> None:
        """Collect GPU metrics if available."""
        
        try:
            import jax
            
            # JAX device information
            devices = jax.devices()
            for i, device in enumerate(devices):
                if device.device_kind == 'gpu':
                    # GPU memory usage (approximation)
                    try:
                        # This is a simplified approach
                        self.metrics_collector.record_metric(
                            f"gpu.{i}.available", 1.0,
                            labels={'device_id': str(i), 'device_kind': device.device_kind}
                        )
                    except Exception as e:
                        self.logger.debug(f"Failed to collect GPU metrics: {e}")
                        
        except ImportError:
            pass
    
    def _collect_simulation_metrics(self) -> None:
        """Collect simulation-specific metrics."""
        
        # These would be updated by simulation components
        # For now, record placeholder metrics
        
        simulation_metrics = {
            "simulation.active_sessions": 0,
            "simulation.total_computations": 0,
            "simulation.average_execution_time": 0.0,
            "simulation.memory_pool_usage": 0.0
        }
        
        for metric_name, value in simulation_metrics.items():
            self.metrics_collector.record_metric(metric_name, value)
    
    def _run_anomaly_detection(self) -> None:
        """Run anomaly detection on collected metrics."""
        
        # Define critical metrics to monitor
        critical_metrics = [
            "system.cpu.usage_percent",
            "system.memory.usage_percent", 
            "system.disk.usage_percent",
            "simulation.average_execution_time"
        ]
        
        for metric_name in critical_metrics:
            # Get recent data
            recent_points = self.metrics_collector.get_metric_history(
                metric_name, time_window=3600  # 1 hour
            )
            
            if len(recent_points) < 10:
                continue
                
            values = [p.value for p in recent_points]
            current_value = values[-1]
            
            # Update baseline
            self.anomaly_detector.update_baseline(metric_name, values[:-1])
            
            # Detect anomaly
            is_anomaly, confidence, reason = self.anomaly_detector.detect_anomaly(
                metric_name, current_value, values[-10:]
            )
            
            if is_anomaly:
                # Determine severity based on metric and confidence
                if "cpu" in metric_name or "memory" in metric_name:
                    severity = AlertSeverity.CRITICAL if confidence > 0.8 else AlertSeverity.WARNING
                else:
                    severity = AlertSeverity.WARNING
                
                # Trigger alert
                self.alert_manager.trigger_alert(
                    alert_id=f"anomaly_{metric_name}",
                    severity=severity,
                    message=f"Anomaly detected in {metric_name}: {current_value:.2f} (confidence: {confidence:.2f})",
                    source="anomaly_detector",
                    metadata={
                        'metric_name': metric_name,
                        'current_value': current_value,
                        'confidence': confidence,
                        'reason': reason
                    }
                )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Determine overall health
        if any(a.severity == AlertSeverity.EMERGENCY for a in active_alerts):
            overall_status = "EMERGENCY"
        elif any(a.severity == AlertSeverity.CRITICAL for a in active_alerts):
            overall_status = "CRITICAL"
        elif any(a.severity == AlertSeverity.WARNING for a in active_alerts):
            overall_status = "WARNING"
        else:
            overall_status = "HEALTHY"
        
        # Get recent metrics
        recent_metrics = {}
        for metric_name in ["system.cpu.usage_percent", "system.memory.usage_percent"]:
            stats = self.metrics_collector.compute_statistics(metric_name, time_window=300)  # 5 min
            recent_metrics[metric_name] = stats
        
        return {
            'overall_status': overall_status,
            'active_alerts_count': len(active_alerts),
            'active_alerts': [
                {
                    'id': a.alert_id,
                    'severity': a.severity.value,
                    'message': a.message,
                    'timestamp': a.timestamp
                } for a in active_alerts[:10]  # Show top 10
            ],
            'recent_metrics': recent_metrics,
            'monitoring_uptime': time.time() - getattr(self, '_start_time', time.time())
        }


def create_monitoring_system(collection_interval: float = 5.0) -> SystemHealthMonitor:
    """Factory function to create monitoring system."""
    
    logger = get_logger(__name__)
    logger.info("Creating intelligent monitoring system")
    
    monitor = SystemHealthMonitor(
        collection_interval=collection_interval,
        enable_gpu_monitoring=True
    )
    
    # Start monitoring
    monitor.start_monitoring()
    
    logger.info("Monitoring system created and started")
    return monitor


# Context manager for monitoring simulation runs
class SimulationMonitor:
    """Context manager for monitoring specific simulation runs."""
    
    def __init__(self, 
                 simulation_name: str,
                 health_monitor: SystemHealthMonitor):
        self.simulation_name = simulation_name
        self.health_monitor = health_monitor
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        
        # Record simulation start
        self.health_monitor.metrics_collector.record_metric(
            "simulation.runs_started", 1, MetricType.COUNTER,
            labels={'simulation_name': self.simulation_name}
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            
            # Record completion metrics
            self.health_monitor.metrics_collector.record_metric(
                "simulation.execution_time", duration, MetricType.TIMER,
                labels={'simulation_name': self.simulation_name}
            )
            
            if exc_type is None:
                # Successful completion
                self.health_monitor.metrics_collector.record_metric(
                    "simulation.runs_completed", 1, MetricType.COUNTER,
                    labels={'simulation_name': self.simulation_name, 'status': 'success'}
                )
            else:
                # Failed execution
                self.health_monitor.metrics_collector.record_metric(
                    "simulation.runs_completed", 1, MetricType.COUNTER,
                    labels={'simulation_name': self.simulation_name, 'status': 'failed'}
                )
                
                # Trigger alert for simulation failure
                self.health_monitor.alert_manager.trigger_alert(
                    alert_id=f"simulation_failure_{self.simulation_name}",
                    severity=AlertSeverity.WARNING,
                    message=f"Simulation {self.simulation_name} failed: {exc_type.__name__}",
                    source="simulation_monitor",
                    metadata={
                        'simulation_name': self.simulation_name,
                        'exception_type': exc_type.__name__,
                        'exception_message': str(exc_val)
                    }
                )


if __name__ == "__main__":
    # Test monitoring system
    monitor = create_monitoring_system()
    
    # Test simulation monitoring
    with SimulationMonitor("test_simulation", monitor):
        time.sleep(2)  # Simulate work
        print("Simulation completed")
    
    # Check health status
    health_status = monitor.get_health_status()
    print(f"System health: {health_status['overall_status']}")
    print(f"Active alerts: {health_status['active_alerts_count']}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    print("Intelligent Monitoring System Implementation Complete")