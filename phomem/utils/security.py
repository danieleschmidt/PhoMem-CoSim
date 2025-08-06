"""
Security measures and input sanitization for PhoMem-CoSim.
"""

import os
import logging
import hashlib
import hmac
import secrets
import time
import re
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import warnings
import json
import base64

import numpy as np
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Security-related errors."""
    def __init__(self, message: str, context: Optional[Dict] = None):
        super().__init__(message)
        self.context = context or {}


class SecurityWarning(UserWarning):
    """Warning for potential security issues."""
    pass


class SecurityValidator:
    """Security validation and hardening utilities."""
    
    def __init__(self):
        self.logger = logger
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.allowed_extensions = {'.py', '.json', '.yaml', '.yml', '.txt', '.md', '.cir', '.sp'}
        self.dangerous_patterns = [
            'exec(', 'eval(', '__import__', 
            'subprocess', 'os.system', 'os.popen',
            'open(', 'file(', 'input(',
            'raw_input(', 'compile('
        ]
    
    def validate_file_path(self, file_path: Union[str, Path], 
                          allowed_dirs: List[str] = None) -> Path:
        """Validate file path for security."""
        path = Path(file_path).resolve()
        
        # Check for path traversal
        if '..' in str(path) or str(path).startswith('/'):
            if not str(path).startswith(('/tmp', '/var/tmp', str(Path.home()))):
                raise SecurityError(
                    f"Potentially unsafe file path: {path}",
                    context={'path': str(path)}
                )
        
        # Check allowed directories
        if allowed_dirs:
            path_str = str(path)
            if not any(path_str.startswith(allowed_dir) for allowed_dir in allowed_dirs):
                raise SecurityError(
                    f"File path outside allowed directories: {path}",
                    context={'path': str(path), 'allowed_dirs': allowed_dirs}
                )
        
        # Check file extension
        if path.suffix.lower() not in self.allowed_extensions:
            self.logger.warning(f"Unusual file extension: {path.suffix}")
        
        return path
    
    def validate_file_content(self, content: str, file_type: str = 'unknown') -> bool:
        """Validate file content for dangerous patterns."""
        content_lower = content.lower()
        
        dangerous_found = []
        for pattern in self.dangerous_patterns:
            if pattern in content_lower:
                dangerous_found.append(pattern)
        
        if dangerous_found:
            raise SecurityError(
                f"Potentially dangerous patterns found in {file_type} content",
                context={'patterns': dangerous_found},
                suggestions=[
                    "Review code for security implications",
                    "Use safe alternatives to dynamic execution",
                    "Validate all user inputs"
                ]
            )
        
        return True
    
    def validate_file_size(self, file_path: Union[str, Path]) -> bool:
        """Validate file size."""
        path = Path(file_path)
        if not path.exists():
            return True
        
        size = path.stat().st_size
        if size > self.max_file_size:
            raise SecurityError(
                f"File too large: {size} bytes (max: {self.max_file_size})",
                context={'file_size': size, 'max_size': self.max_file_size}
            )
        
        return True
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent issues."""
        # Remove dangerous characters
        dangerous_chars = '<>:"/\\|?*'
        sanitized = ''.join(c for c in filename if c not in dangerous_chars)
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        
        # Limit length
        if len(sanitized) > 255:
            sanitized = sanitized[:255]
        
        # Ensure not empty
        if not sanitized:
            sanitized = 'unnamed_file'
        
        return sanitized
    
    def create_secure_temp_file(self, suffix: str = '', prefix: str = 'phomem_') -> str:
        """Create secure temporary file."""
        # Use secure random for filename
        random_part = secrets.token_hex(8)
        filename = f"{prefix}{random_part}{suffix}"
        
        # Create in secure temp directory
        temp_dir = Path(tempfile.gettempdir()) / 'phomem_secure'
        temp_dir.mkdir(mode=0o700, exist_ok=True)  # Owner-only permissions
        
        temp_file = temp_dir / filename
        temp_file.touch(mode=0o600)  # Owner read/write only
        
        self.logger.debug(f"Created secure temp file: {temp_file}")
        return str(temp_file)
    
    def hash_sensitive_data(self, data: str, algorithm: str = 'sha256') -> str:
        """Hash sensitive data for logging/storage."""
        if algorithm == 'sha256':
            hasher = hashlib.sha256()
        elif algorithm == 'sha512':
            hasher = hashlib.sha512()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        hasher.update(data.encode('utf-8'))
        return hasher.hexdigest()
    
    def redact_sensitive_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive information from data structures."""
        sensitive_keys = {
            'password', 'passwd', 'secret', 'token', 'key', 'auth',
            'credential', 'private', 'confidential'
        }
        
        redacted = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                if isinstance(value, str):
                    redacted[key] = f"[REDACTED:{len(value)} chars]"
                else:
                    redacted[key] = "[REDACTED]"
            elif isinstance(value, dict):
                redacted[key] = self.redact_sensitive_info(value)
            elif isinstance(value, list):
                redacted[key] = [
                    self.redact_sensitive_info(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                redacted[key] = value
        
        return redacted


class ConfigurationSecurity:
    """Security for configuration files and parameters."""
    
    def __init__(self):
        self.logger = get_logger('security.config')
        self.validator = SecurityValidator()
    
    def validate_config_file(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate and load configuration file securely."""
        path = self.validator.validate_file_path(config_path)
        self.validator.validate_file_size(path)
        
        try:
            with open(path, 'r') as f:
                content = f.read()
            
            # Validate content
            self.validator.validate_file_content(content, 'configuration')
            
            # Parse configuration
            if path.suffix.lower() in ['.json']:
                config = json.loads(content)
            elif path.suffix.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    config = yaml.safe_load(content)
                except ImportError:
                    raise ConfigurationError(
                        "YAML support not available",
                        suggestions=["Install PyYAML: pip install PyYAML"]
                    )
            else:
                raise ConfigurationError(f"Unsupported configuration format: {path.suffix}")
            
            # Validate configuration structure
            self.validate_config_structure(config)
            
            self.logger.info(f"Successfully validated config: {path}")
            return config
            
        except Exception as e:
            if isinstance(e, (SecurityError, ConfigurationError)):
                raise
            raise ConfigurationError(
                f"Failed to load configuration: {str(e)}",
                config_file=str(path)
            )
    
    def validate_config_structure(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure."""
        required_sections = ['simulation', 'devices', 'logging']
        
        for section in required_sections:
            if section not in config:
                self.logger.warning(f"Missing configuration section: {section}")
        
        # Validate simulation parameters
        if 'simulation' in config:
            sim_config = config['simulation']
            
            # Check for reasonable values
            if 'max_iterations' in sim_config:
                max_iter = sim_config['max_iterations']
                if not isinstance(max_iter, int) or max_iter < 1 or max_iter > 1000000:
                    raise ConfigurationError(
                        f"Invalid max_iterations: {max_iter}",
                        config_section='simulation',
                        suggestions=["Use value between 1 and 1,000,000"]
                    )
            
            if 'convergence_tolerance' in sim_config:
                tol = sim_config['convergence_tolerance']
                if not isinstance(tol, (int, float)) or tol <= 0 or tol >= 1:
                    raise ConfigurationError(
                        f"Invalid convergence_tolerance: {tol}",
                        config_section='simulation',
                        suggestions=["Use positive value less than 1.0"]
                    )
        
        return True
    
    def sanitize_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameters for safe usage."""
        sanitized = {}
        
        for key, value in params.items():
            # Sanitize key
            safe_key = self.validator.sanitize_filename(key).replace(' ', '_')
            
            # Sanitize value based on type
            if isinstance(value, str):
                # Remove potentially dangerous characters
                safe_value = value.replace('\x00', '').strip()
                if len(safe_value) > 10000:  # Limit string length
                    safe_value = safe_value[:10000]
                sanitized[safe_key] = safe_value
            elif isinstance(value, (int, float)):
                # Check for reasonable numeric ranges
                if abs(value) > 1e12:
                    self.logger.warning(f"Large numeric value in {key}: {value}")
                sanitized[safe_key] = value
            elif isinstance(value, dict):
                sanitized[safe_key] = self.sanitize_parameters(value)
            elif isinstance(value, list):
                if len(value) > 10000:  # Limit list size
                    self.logger.warning(f"Large list in {key}: {len(value)} items")
                    value = value[:10000]
                sanitized[safe_key] = value
            else:
                sanitized[safe_key] = value
        
        return sanitized


class InputSanitizer:
    """Sanitize user inputs and simulation parameters."""
    
    def __init__(self):
        self.logger = get_logger('security.input')
    
    def sanitize_array_input(self, array_data: Any, name: str) -> Any:
        """Sanitize array inputs."""
        # This would typically use JAX/NumPy validation
        # For now, basic checks
        try:
            if hasattr(array_data, 'shape'):
                # Check for reasonable array sizes
                total_elements = 1
                for dim in array_data.shape:
                    total_elements *= dim
                
                if total_elements > 1e9:  # 1 billion elements
                    raise SecurityError(
                        f"Array {name} too large: {total_elements} elements",
                        context={'array_shape': array_data.shape}
                    )
                
                # Check for NaN/inf (potential attack vector)
                try:
                    if hasattr(array_data, 'dtype') and 'float' in str(array_data.dtype):
                        import numpy as np
                        if np.any(np.isnan(array_data)) or np.any(np.isinf(array_data)):
                            raise SecurityError(
                                f"Array {name} contains NaN or infinite values",
                                suggestions=["Validate input data source"]
                            )
                except ImportError:
                    pass  # NumPy not available
            
            return array_data
            
        except Exception as e:
            if isinstance(e, SecurityError):
                raise
            raise SecurityError(f"Failed to validate array input {name}: {str(e)}")
    
    def sanitize_file_upload(self, file_content: bytes, filename: str) -> Tuple[bytes, str]:
        """Sanitize uploaded file content."""
        validator = SecurityValidator()
        
        # Sanitize filename
        safe_filename = validator.sanitize_filename(filename)
        
        # Check file size
        if len(file_content) > validator.max_file_size:
            raise SecurityError(
                f"File too large: {len(file_content)} bytes",
                context={'max_size': validator.max_file_size}
            )
        
        # Basic content validation
        try:
            content_str = file_content.decode('utf-8', errors='ignore')
            validator.validate_file_content(content_str, 'uploaded file')
        except UnicodeDecodeError:
            # Binary file - additional checks could be added
            self.logger.warning(f"Binary file uploaded: {safe_filename}")
        
        return file_content, safe_filename


# Global instances
_security_validator = None
_config_security = None
_input_sanitizer = None

def get_security_validator() -> SecurityValidator:
    """Get global security validator instance."""
    global _security_validator
    if _security_validator is None:
        _security_validator = SecurityValidator()
    return _security_validator

def get_config_security() -> ConfigurationSecurity:
    """Get global configuration security instance."""
    global _config_security
    if _config_security is None:
        _config_security = ConfigurationSecurity()
    return _config_security

def get_input_sanitizer() -> InputSanitizer:
    """Get global input sanitizer instance."""
    global _input_sanitizer
    if _input_sanitizer is None:
        _input_sanitizer = InputSanitizer()
    return _input_sanitizer