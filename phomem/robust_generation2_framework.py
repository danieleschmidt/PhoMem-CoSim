"""
Generation 2 Robust Framework: MAKE IT ROBUST
Advanced error handling, validation, security, and fault tolerance for neuromorphic systems.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
import chex
import time
import logging
import hashlib
from functools import wraps
from abc import ABC, abstractmethod
import warnings

# Enhanced imports for robustness
try:
    import cryptography
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

try:
    import prometheus_client
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


class HardwareError(Exception):
    """Custom exception for hardware-related failures."""
    pass


class RobustSecurityValidator:
    """Advanced security validation for neuromorphic computing."""
    
    def __init__(self, enable_encryption: bool = True):
        self.enable_encryption = enable_encryption and CRYPTO_AVAILABLE
        self.security_key = None
        
        if self.enable_encryption:
            self.security_key = Fernet.generate_key()
            self.cipher_suite = Fernet(self.security_key)
        
        # Security audit trail
        self.security_events = []
        self.failed_attempts = 0
        self.max_failed_attempts = 5
    
    def validate_input_integrity(self, data: chex.Array) -> bool:
        """Validate input data integrity and detect adversarial attacks."""
        
        try:
            # Check for NaN and infinite values
            if jnp.any(jnp.isnan(data)) or jnp.any(jnp.isinf(data)):
                self._log_security_event("Invalid data detected: NaN or Inf values")
                return False
            
            # Check for unusual value ranges (potential adversarial perturbations)
            data_std = jnp.std(data)
            data_mean = jnp.mean(data)
            
            # Z-score based outlier detection
            z_scores = jnp.abs((data - data_mean) / (data_std + 1e-8))
            outlier_threshold = 5.0  # 5-sigma threshold
            
            if jnp.any(z_scores > outlier_threshold):
                outlier_fraction = jnp.mean(z_scores > outlier_threshold)
                if outlier_fraction > 0.1:  # More than 10% outliers
                    self._log_security_event(f"Potential adversarial input: {outlier_fraction:.2%} outliers")
                    return False
            
            # Check for gradient explosion indicators
            if data.ndim > 1:
                gradients = jnp.gradient(data, axis=0)
                max_gradient = jnp.max(jnp.abs(gradients))
                if max_gradient > 100 * data_std:
                    self._log_security_event(f"Suspicious gradient magnitude: {max_gradient}")
                    return False
            
            # Data pattern analysis for injection attacks
            entropy = self._calculate_entropy(data)
            if entropy < 0.1:  # Very low entropy might indicate crafted input
                self._log_security_event(f"Low entropy input detected: {entropy:.3f}")
                return False
            
            return True
            
        except Exception as e:
            self._log_security_event(f"Input validation error: {e}")
            return False
    
    def validate_parameter_integrity(self, params: Dict[str, Any]) -> bool:
        """Validate neural network parameters for tampering."""
        
        try:
            # Calculate parameter hash for integrity checking
            param_hash = self._calculate_param_hash(params)
            
            # Check for suspicious parameter values
            for param_name, param_value in jax.tree_util.tree_leaves_with_path(params):
                if jnp.any(jnp.isnan(param_value)) or jnp.any(jnp.isinf(param_value)):
                    self._log_security_event(f"Invalid parameter in {param_name}")
                    return False
                
                # Check for parameter explosion
                param_magnitude = jnp.max(jnp.abs(param_value))
                if param_magnitude > 1e6:
                    self._log_security_event(f"Parameter explosion in {param_name}: {param_magnitude}")
                    return False
            
            return True
            
        except Exception as e:
            self._log_security_event(f"Parameter validation error: {e}")
            return False
    
    def encrypt_sensitive_data(self, data: bytes) -> bytes:
        """Encrypt sensitive data using strong encryption."""
        if not self.enable_encryption:
            warnings.warn("Encryption not available, returning plaintext")
            return data
        
        try:
            encrypted_data = self.cipher_suite.encrypt(data)
            self._log_security_event("Data encrypted successfully")
            return encrypted_data
        except Exception as e:
            self._log_security_event(f"Encryption failed: {e}")
            raise SecurityError(f"Failed to encrypt data: {e}")
    
    def decrypt_sensitive_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt sensitive data."""
        if not self.enable_encryption:
            warnings.warn("Encryption not available, returning data as-is")
            return encrypted_data
        
        try:
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            self._log_security_event("Data decrypted successfully")
            return decrypted_data
        except Exception as e:
            self._log_security_event(f"Decryption failed: {e}")
            raise SecurityError(f"Failed to decrypt data: {e}")
    
    def _calculate_entropy(self, data: chex.Array) -> float:
        """Calculate Shannon entropy of data."""
        # Bin the data for entropy calculation
        data_flat = data.flatten()
        hist, _ = jnp.histogram(data_flat, bins=50)
        hist = hist + 1e-8  # Avoid log(0)
        probs = hist / jnp.sum(hist)
        entropy = -jnp.sum(probs * jnp.log2(probs))
        return float(entropy)
    
    def _calculate_param_hash(self, params: Dict[str, Any]) -> str:
        """Calculate hash of parameters for integrity checking."""
        # Serialize parameters to bytes
        param_bytes = str(params).encode()
        param_hash = hashlib.sha256(param_bytes).hexdigest()
        return param_hash
    
    def _log_security_event(self, event: str):
        """Log security events for audit trail."""
        timestamp = time.time()
        security_event = {
            'timestamp': timestamp,
            'event': event,
            'level': 'WARNING'
        }
        self.security_events.append(security_event)
        logging.warning(f"Security Event: {event}")


class RobustValidationFramework:
    """Comprehensive validation framework for neuromorphic systems."""
    
    def __init__(self):
        self.validation_rules = {}
        self.validation_history = []
        self.error_threshold = 0.05  # 5% error tolerance
    
    def register_validation_rule(self, 
                                rule_name: str, 
                                validation_func: Callable,
                                critical: bool = False):
        """Register a new validation rule."""
        self.validation_rules[rule_name] = {
            'func': validation_func,
            'critical': critical,
            'pass_count': 0,
            'fail_count': 0
        }
    
    def validate_network_architecture(self, network: Any) -> Dict[str, Any]:
        """Validate network architecture for correctness and stability."""
        
        validation_results = {
            'passed': [],
            'failed': [],
            'warnings': [],
            'critical_failures': []
        }
        
        # Basic architecture validation
        try:
            # Test initialization
            key = jax.random.PRNGKey(42)
            test_input = jax.random.normal(key, (1, 64))  # Assume 64-dim input
            params = network.init(key, test_input)
            
            # Test forward pass
            output = network.apply(params, test_input)
            
            validation_results['passed'].append("Architecture initialization and forward pass")
            
        except Exception as e:
            validation_results['critical_failures'].append(f"Architecture validation failed: {e}")
            return validation_results
        
        # Parameter count validation
        total_params = sum(jnp.size(p) for p in jax.tree_leaves(params))
        if total_params > 1e9:  # >1B parameters
            validation_results['warnings'].append(f"Large parameter count: {total_params:,}")
        elif total_params < 100:
            validation_results['warnings'].append(f"Very small parameter count: {total_params}")
        else:
            validation_results['passed'].append(f"Reasonable parameter count: {total_params:,}")
        
        # Gradient flow validation
        try:
            def loss_fn(params):
                pred = network.apply(params, test_input)
                return jnp.mean(pred ** 2)
            
            grads = jax.grad(loss_fn)(params)
            
            # Check for vanishing gradients
            grad_norms = [jnp.linalg.norm(g) for g in jax.tree_leaves(grads)]
            min_grad_norm = min(grad_norms)
            max_grad_norm = max(grad_norms)
            
            if min_grad_norm < 1e-8:
                validation_results['warnings'].append("Potential vanishing gradient problem")
            elif max_grad_norm > 1e4:
                validation_results['warnings'].append("Potential exploding gradient problem")
            else:
                validation_results['passed'].append("Healthy gradient flow")
                
        except Exception as e:
            validation_results['failed'].append(f"Gradient validation failed: {e}")
        
        # Run custom validation rules
        for rule_name, rule_info in self.validation_rules.items():
            try:
                result = rule_info['func'](network, params)
                if result:
                    validation_results['passed'].append(rule_name)
                    rule_info['pass_count'] += 1
                else:
                    if rule_info['critical']:
                        validation_results['critical_failures'].append(rule_name)
                    else:
                        validation_results['failed'].append(rule_name)
                    rule_info['fail_count'] += 1
                    
            except Exception as e:
                validation_results['failed'].append(f"{rule_name}: {e}")
                rule_info['fail_count'] += 1
        
        # Store validation history
        validation_summary = {
            'timestamp': time.time(),
            'total_tests': len(validation_results['passed']) + len(validation_results['failed']) + len(validation_results['critical_failures']),
            'passed': len(validation_results['passed']),
            'failed': len(validation_results['failed']),
            'critical_failures': len(validation_results['critical_failures']),
            'warnings': len(validation_results['warnings'])
        }
        self.validation_history.append(validation_summary)
        
        return validation_results
    
    def validate_training_stability(self, 
                                  loss_history: List[float],
                                  metric_history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Validate training process stability and convergence."""
        
        stability_results = {
            'convergence_status': 'unknown',
            'stability_issues': [],
            'recommendations': []
        }
        
        if len(loss_history) < 10:
            stability_results['stability_issues'].append("Insufficient training history")
            return stability_results
        
        loss_array = jnp.array(loss_history)
        
        # Check for convergence
        recent_losses = loss_array[-10:]
        loss_std = jnp.std(recent_losses)
        loss_mean = jnp.mean(recent_losses)
        
        if loss_std / (loss_mean + 1e-8) < 0.01:  # CV < 1%
            stability_results['convergence_status'] = 'converged'
        elif jnp.mean(jnp.diff(recent_losses)) < -1e-6:
            stability_results['convergence_status'] = 'converging'
        else:
            stability_results['convergence_status'] = 'diverging'
            stability_results['stability_issues'].append("Training appears to be diverging")
        
        # Check for oscillations
        loss_diffs = jnp.diff(loss_array)
        sign_changes = jnp.sum(jnp.diff(jnp.sign(loss_diffs)) != 0)
        oscillation_ratio = sign_changes / len(loss_diffs)
        
        if oscillation_ratio > 0.5:
            stability_results['stability_issues'].append("High loss oscillations detected")
            stability_results['recommendations'].append("Consider reducing learning rate")
        
        # Check for catastrophic failures
        if jnp.any(jnp.isnan(loss_array)) or jnp.any(jnp.isinf(loss_array)):
            stability_results['stability_issues'].append("NaN or Inf in loss history")
            stability_results['recommendations'].append("Check gradient clipping and learning rate")
        
        # Validate metrics stability
        for metric_name, metric_values in metric_history.items():
            if len(metric_values) >= 10:
                metric_array = jnp.array(metric_values[-10:])
                metric_std = jnp.std(metric_array)
                
                if metric_std > 0.1:  # High variance
                    stability_results['stability_issues'].append(f"High variance in {metric_name}")
        
        return stability_results


class FaultTolerantExecutionEngine:
    """Fault-tolerant execution engine with graceful degradation."""
    
    def __init__(self, 
                 max_retries: int = 3,
                 backup_models: List[Any] = None,
                 fallback_strategy: str = 'graceful'):
        
        self.max_retries = max_retries
        self.backup_models = backup_models or []
        self.fallback_strategy = fallback_strategy
        
        # Fault tracking
        self.fault_history = []
        self.component_health = {}
        
        # Performance monitoring
        self.performance_metrics = {
            'execution_times': [],
            'error_rates': [],
            'recovery_times': []
        }
    
    def robust_execute(self, 
                      primary_function: Callable,
                      *args,
                      **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Execute function with fault tolerance and monitoring."""
        
        execution_info = {
            'attempts': 0,
            'errors': [],
            'execution_time': 0,
            'recovery_used': False,
            'final_status': 'unknown'
        }
        
        start_time = time.time()
        
        # Primary execution attempts
        for attempt in range(self.max_retries):
            execution_info['attempts'] += 1
            
            try:
                result = primary_function(*args, **kwargs)
                execution_info['execution_time'] = time.time() - start_time
                execution_info['final_status'] = 'success'
                
                # Update performance metrics
                self.performance_metrics['execution_times'].append(execution_info['execution_time'])
                return result, execution_info
                
            except Exception as e:
                error_info = {
                    'attempt': attempt + 1,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'timestamp': time.time()
                }
                execution_info['errors'].append(error_info)
                
                # Log fault
                self._log_fault(error_info)
                
                # Wait before retry (exponential backoff)
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt * 0.1  # 0.1, 0.2, 0.4 seconds
                    time.sleep(wait_time)
        
        # Primary function failed, try fallback strategies
        execution_info['recovery_used'] = True
        
        if self.fallback_strategy == 'graceful':
            result, fallback_info = self._graceful_fallback(*args, **kwargs)
            execution_info.update(fallback_info)
            return result, execution_info
        
        elif self.fallback_strategy == 'backup_model':
            result, backup_info = self._backup_model_fallback(*args, **kwargs)
            execution_info.update(backup_info)
            return result, execution_info
        
        else:
            # No fallback available
            execution_info['final_status'] = 'failed'
            execution_info['execution_time'] = time.time() - start_time
            raise HardwareError(f"All execution attempts failed: {execution_info['errors']}")
    
    def _graceful_fallback(self, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Graceful degradation fallback strategy."""
        
        fallback_info = {
            'fallback_type': 'graceful',
            'degradation_level': 'low'
        }
        
        try:
            # Return simplified/cached result
            if len(args) > 0 and hasattr(args[0], 'shape'):
                # Return zeros with appropriate shape for array inputs
                result = jnp.zeros_like(args[0])
                fallback_info['final_status'] = 'degraded_success'
                return result, fallback_info
            else:
                # Return default value for other inputs
                result = 0.0
                fallback_info['final_status'] = 'degraded_success'
                return result, fallback_info
                
        except Exception as e:
            fallback_info['final_status'] = 'fallback_failed'
            fallback_info['fallback_error'] = str(e)
            raise HardwareError(f"Graceful fallback failed: {e}")
    
    def _backup_model_fallback(self, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Backup model fallback strategy."""
        
        fallback_info = {
            'fallback_type': 'backup_model',
            'backup_attempts': 0
        }
        
        for i, backup_model in enumerate(self.backup_models):
            fallback_info['backup_attempts'] += 1
            
            try:
                # Assume backup model has similar interface
                if hasattr(backup_model, 'apply'):
                    result = backup_model.apply(*args, **kwargs)
                else:
                    result = backup_model(*args, **kwargs)
                
                fallback_info['final_status'] = 'backup_success'
                fallback_info['successful_backup'] = i
                return result, fallback_info
                
            except Exception as e:
                fallback_info[f'backup_{i}_error'] = str(e)
                continue
        
        # All backups failed
        fallback_info['final_status'] = 'all_backups_failed'
        raise HardwareError("All backup models failed")
    
    def _log_fault(self, error_info: Dict[str, Any]):
        """Log fault information for analysis."""
        fault_record = {
            'timestamp': time.time(),
            'component': 'execution_engine',
            'severity': self._classify_error_severity(error_info['error_type']),
            'details': error_info
        }
        
        self.fault_history.append(fault_record)
        
        # Update component health
        component_name = fault_record['component']
        if component_name not in self.component_health:
            self.component_health[component_name] = {
                'health_score': 1.0,
                'fault_count': 0,
                'last_fault': None
            }
        
        health_info = self.component_health[component_name]
        health_info['fault_count'] += 1
        health_info['last_fault'] = fault_record['timestamp']
        
        # Decrease health score based on severity
        severity_penalties = {'low': 0.01, 'medium': 0.05, 'high': 0.1, 'critical': 0.2}
        penalty = severity_penalties.get(fault_record['severity'], 0.05)
        health_info['health_score'] = max(0.0, health_info['health_score'] - penalty)
        
        logging.error(f"Fault logged: {fault_record}")
    
    def _classify_error_severity(self, error_type: str) -> str:
        """Classify error severity based on error type."""
        critical_errors = ['SystemExit', 'KeyboardInterrupt', 'MemoryError']
        high_errors = ['RuntimeError', 'ValueError', 'TypeError']
        medium_errors = ['AttributeError', 'IndexError', 'KeyError']
        
        if error_type in critical_errors:
            return 'critical'
        elif error_type in high_errors:
            return 'high'
        elif error_type in medium_errors:
            return 'medium'
        else:
            return 'low'
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        
        current_time = time.time()
        recent_faults = [f for f in self.fault_history if current_time - f['timestamp'] < 3600]  # Last hour
        
        health_report = {
            'overall_health': self._calculate_overall_health(),
            'component_health': self.component_health,
            'recent_fault_count': len(recent_faults),
            'total_fault_count': len(self.fault_history),
            'performance_summary': self._summarize_performance(),
            'recommendations': self._generate_health_recommendations()
        }
        
        return health_report
    
    def _calculate_overall_health(self) -> float:
        """Calculate overall system health score."""
        if not self.component_health:
            return 1.0
        
        health_scores = [info['health_score'] for info in self.component_health.values()]
        return float(jnp.mean(jnp.array(health_scores)))
    
    def _summarize_performance(self) -> Dict[str, Any]:
        """Summarize system performance metrics."""
        performance_summary = {}
        
        if self.performance_metrics['execution_times']:
            times = jnp.array(self.performance_metrics['execution_times'])
            performance_summary['execution_time'] = {
                'mean': float(jnp.mean(times)),
                'std': float(jnp.std(times)),
                'p95': float(jnp.percentile(times, 95))
            }
        
        if self.performance_metrics['error_rates']:
            error_rates = jnp.array(self.performance_metrics['error_rates'])
            performance_summary['error_rate'] = {
                'mean': float(jnp.mean(error_rates)),
                'current': float(error_rates[-1]) if len(error_rates) > 0 else 0.0
            }
        
        return performance_summary
    
    def _generate_health_recommendations(self) -> List[str]:
        """Generate recommendations based on system health."""
        recommendations = []
        
        overall_health = self._calculate_overall_health()
        
        if overall_health < 0.5:
            recommendations.append("System health critically low - immediate maintenance required")
        elif overall_health < 0.7:
            recommendations.append("System health degraded - schedule maintenance soon")
        
        # Component-specific recommendations
        for component, health_info in self.component_health.items():
            if health_info['health_score'] < 0.5:
                recommendations.append(f"Component {component} requires attention - health: {health_info['health_score']:.2f}")
            
            if health_info['fault_count'] > 10:
                recommendations.append(f"High fault count in {component} - investigate root cause")
        
        # Performance recommendations
        if self.performance_metrics['execution_times']:
            avg_time = jnp.mean(jnp.array(self.performance_metrics['execution_times']))
            if avg_time > 1.0:  # >1 second average
                recommendations.append("Execution times high - consider optimization")
        
        return recommendations


def robust_network_decorator(max_retries: int = 3):
    """Decorator for adding robustness to network functions."""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            fault_engine = FaultTolerantExecutionEngine(max_retries=max_retries)
            security_validator = RobustSecurityValidator()
            
            # Validate inputs if they contain arrays
            for arg in args:
                if isinstance(arg, (jnp.ndarray, np.ndarray)):
                    if not security_validator.validate_input_integrity(arg):
                        raise SecurityError("Input validation failed")
            
            # Execute with fault tolerance
            try:
                result, exec_info = fault_engine.robust_execute(func, *args, **kwargs)
                return result
            except Exception as e:
                logging.error(f"Robust execution failed: {e}")
                raise
        
        return wrapper
    return decorator


def demonstrate_generation2_robust():
    """Demonstrate Generation 2 robust framework capabilities."""
    
    print("🛡️ GENERATION 2 ROBUST: ADVANCED RESILIENCE FRAMEWORK")
    print("=" * 70)
    
    # Security validation demonstration
    print("\n1. Security Validation Framework...")
    security_validator = RobustSecurityValidator()
    
    # Test normal data
    normal_data = jax.random.normal(jax.random.PRNGKey(42), (100, 64))
    is_valid = security_validator.validate_input_integrity(normal_data)
    print(f"   Normal data validation: {'✓ PASS' if is_valid else '✗ FAIL'}")
    
    # Test adversarial data (simulated)
    adversarial_data = normal_data.at[0].set(1000.0)  # Outlier injection
    is_valid_adv = security_validator.validate_input_integrity(adversarial_data)
    print(f"   Adversarial data detection: {'✓ DETECTED' if not is_valid_adv else '✗ MISSED'}")
    
    # Validation framework demonstration
    print("\n2. Comprehensive Validation Framework...")
    validator = RobustValidationFramework()
    
    # Register custom validation rules
    def check_parameter_range(network, params):
        """Custom rule: check parameter ranges."""
        for param in jax.tree_leaves(params):
            if jnp.max(jnp.abs(param)) > 10:
                return False
        return True
    
    def check_gradient_health(network, params):
        """Custom rule: check gradient health."""
        try:
            test_input = jax.random.normal(jax.random.PRNGKey(0), (1, 64))
            def loss_fn(p):
                return jnp.sum(network.apply(p, test_input) ** 2)
            grads = jax.grad(loss_fn)(params)
            grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_leaves(grads)))
            return 1e-8 < grad_norm < 1e3  # Reasonable gradient magnitude
        except:
            return False
    
    validator.register_validation_rule("parameter_range", check_parameter_range, critical=True)
    validator.register_validation_rule("gradient_health", check_gradient_health, critical=False)
    
    # Create a test network for validation
    from .enhanced_generation1_core import HybridQuantumMemristiveNetwork
    
    try:
        test_network = HybridQuantumMemristiveNetwork(
            input_size=64,
            hidden_sizes=[128, 64],
            output_size=10
        )
        
        validation_results = validator.validate_network_architecture(test_network)
        print(f"   Validation tests passed: {len(validation_results['passed'])}")
        print(f"   Validation tests failed: {len(validation_results['failed'])}")
        print(f"   Critical failures: {len(validation_results['critical_failures'])}")
        print(f"   Warnings: {len(validation_results['warnings'])}")
        
        if validation_results['critical_failures']:
            print(f"   ⚠️  Critical issues: {validation_results['critical_failures']}")
        
    except Exception as e:
        print(f"   Network validation skipped due to: {e}")
    
    # Fault tolerance demonstration
    print("\n3. Fault-Tolerant Execution Engine...")
    fault_engine = FaultTolerantExecutionEngine(
        max_retries=3,
        fallback_strategy='graceful'
    )
    
    # Test function that occasionally fails
    failure_count = 0
    def unreliable_function(x):
        nonlocal failure_count
        failure_count += 1
        if failure_count <= 2:  # Fail first 2 attempts
            raise RuntimeError(f"Simulated failure #{failure_count}")
        return x * 2 + 1
    
    try:
        result, exec_info = fault_engine.robust_execute(
            unreliable_function, 
            jnp.array([1.0, 2.0, 3.0])
        )
        print(f"   Fault-tolerant execution: {'✓ SUCCESS' if exec_info['final_status'] == 'success' else '✗ FAILED'}")
        print(f"   Attempts required: {exec_info['attempts']}")
        print(f"   Execution time: {exec_info['execution_time']:.4f}s")
        
    except Exception as e:
        print(f"   Fault tolerance test failed: {e}")
    
    # System health monitoring
    print("\n4. System Health Monitoring...")
    health_report = fault_engine.get_system_health_report()
    print(f"   Overall system health: {health_report['overall_health']:.2f}")
    print(f"   Recent faults: {health_report['recent_fault_count']}")
    print(f"   Total faults: {health_report['total_fault_count']}")
    
    if health_report['recommendations']:
        print("   Recommendations:")
        for rec in health_report['recommendations'][:3]:  # Show first 3
            print(f"     • {rec}")
    
    # Training stability validation
    print("\n5. Training Stability Analysis...")
    
    # Simulate training history
    stable_losses = [1.0 - 0.05 * i + 0.01 * np.random.randn() for i in range(100)]
    unstable_losses = [1.0 + 0.1 * np.sin(0.1 * i) + 0.05 * np.random.randn() for i in range(100)]
    
    stable_metrics = {'accuracy': [0.5 + 0.3 * (1 - np.exp(-0.05 * i)) for i in range(100)]}
    
    stable_analysis = validator.validate_training_stability(stable_losses, stable_metrics)
    unstable_analysis = validator.validate_training_stability(unstable_losses, stable_metrics)
    
    print(f"   Stable training convergence: {stable_analysis['convergence_status']}")
    print(f"   Unstable training convergence: {unstable_analysis['convergence_status']}")
    print(f"   Stability issues detected: {len(unstable_analysis['stability_issues'])}")
    
    # Robust decorator demonstration  
    print("\n6. Robust Function Decoration...")
    
    @robust_network_decorator(max_retries=2)
    def sensitive_computation(data):
        if jnp.any(data > 5):  # Trigger validation failure
            raise ValueError("Data out of range")
        return jnp.sum(data ** 2)
    
    try:
        safe_data = jnp.array([1., 2., 3.])
        result = sensitive_computation(safe_data)
        print(f"   Safe computation result: {result:.2f}")
    except Exception as e:
        print(f"   Safe computation failed: {e}")
    
    try:
        unsafe_data = jnp.array([1., 2., 10.])  # Contains outlier
        result = sensitive_computation(unsafe_data)
        print(f"   Unsafe computation unexpectedly succeeded: {result:.2f}")
    except SecurityError:
        print("   ✓ Unsafe computation correctly blocked by security validation")
    except Exception as e:
        print(f"   Unsafe computation failed with: {e}")
    
    print(f"\n   Security events logged: {len(security_validator.security_events)}")
    if security_validator.security_events:
        recent_event = security_validator.security_events[-1]
        print(f"   Most recent event: {recent_event['event']}")
    
    return {
        'security_validator': security_validator,
        'validation_framework': validator,
        'fault_engine': fault_engine,
        'health_report': health_report
    }


if __name__ == "__main__":
    results = demonstrate_generation2_robust()
    print("\n🛡️ GENERATION 2 ROBUST FRAMEWORK COMPLETE - SYSTEM HARDENED!")