#!/usr/bin/env python3
"""
Simplified quality gates runner for PhoMem-CoSim.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_imports():
    """Test basic imports."""
    print("Testing basic imports...")
    
    results = {}
    
    # Test core imports
    try:
        import phomem
        results['phomem'] = 'SUCCESS'
    except ImportError as e:
        results['phomem'] = f'FAILED: {e}'
    
    try:
        from phomem.config import PhoMemConfig
        config = PhoMemConfig()
        results['config'] = 'SUCCESS'
    except Exception as e:
        results['config'] = f'FAILED: {e}'
    
    try:
        from phomem.utils.data import DataManager
        dm = DataManager()
        results['data_utils'] = 'SUCCESS'
    except Exception as e:
        results['data_utils'] = f'FAILED: {e}'
    
    try:
        from phomem.utils.plotting import plot_network_performance
        results['plotting'] = 'SUCCESS'
    except Exception as e:
        results['plotting'] = f'FAILED: {e}'
    
    try:
        from phomem.utils.performance import get_performance_optimizer
        optimizer = get_performance_optimizer()
        results['performance'] = 'SUCCESS'
    except Exception as e:
        results['performance'] = f'FAILED: {e}'
    
    try:
        from phomem.utils.security import get_security_validator
        validator = get_security_validator()
        results['security'] = 'SUCCESS'
    except Exception as e:
        results['security'] = f'FAILED: {e}'
    
    return results

def test_basic_functionality():
    """Test basic functionality."""
    print("Testing basic functionality...")
    
    results = {}
    
    # Test configuration
    try:
        from phomem.config import PhoMemConfig
        config = PhoMemConfig()
        is_valid = config.validate()
        results['config_validation'] = 'SUCCESS' if is_valid else 'WARNING: Invalid config'
    except Exception as e:
        results['config_validation'] = f'FAILED: {e}'
    
    # Test data utilities
    try:
        from phomem.utils.data import save_simulation_results, load_measurement_data
        test_data = {'test': [1, 2, 3], 'results': {'accuracy': 0.95}}
        save_simulation_results(test_data, '/tmp/test_results.json', format='json')
        loaded = load_measurement_data('/tmp/test_results.json')
        results['data_io'] = 'SUCCESS'
    except Exception as e:
        results['data_io'] = f'FAILED: {e}'
    
    # Test performance utilities
    try:
        from phomem.utils.performance import get_performance_optimizer
        optimizer = get_performance_optimizer()
        
        @optimizer.memoize(ttl=10)
        def test_function(x):
            return x * x
        
        result1 = test_function(5)
        result2 = test_function(5)  # Should be cached
        
        if result1 == result2 == 25:
            results['memoization'] = 'SUCCESS'
        else:
            results['memoization'] = 'FAILED: Incorrect results'
    except Exception as e:
        results['memoization'] = f'FAILED: {e}'
    
    return results

def test_security_features():
    """Test security features."""
    print("Testing security features...")
    
    results = {}
    
    try:
        from phomem.utils.security import get_security_validator
        validator = get_security_validator()
        
        # Test path validation
        try:
            safe_path = validator.validate_file_path('/tmp/safe_file.txt')
            results['path_validation_safe'] = 'SUCCESS'
        except Exception:
            results['path_validation_safe'] = 'FAILED'
        
        # Test dangerous path rejection
        try:
            dangerous_path = validator.validate_file_path('../../../etc/passwd')
            results['path_validation_dangerous'] = 'FAILED: Should have rejected dangerous path'
        except Exception:
            results['path_validation_dangerous'] = 'SUCCESS: Correctly rejected dangerous path'
        
        # Test sensitive data redaction
        test_data = {
            'username': 'testuser',
            'password': 'secret123',
            'normal_field': 'normal_value'
        }
        redacted = validator.redact_sensitive_info(test_data)
        
        if '[REDACTED' in str(redacted.get('password', '')) and 'normal_value' in str(redacted.get('normal_field', '')):
            results['data_redaction'] = 'SUCCESS'
        else:
            results['data_redaction'] = 'FAILED: Redaction not working correctly'
        
    except Exception as e:
        results['security_general'] = f'FAILED: {e}'
    
    return results

def check_dependencies():
    """Check for optional dependencies."""
    print("Checking dependencies...")
    
    results = {}
    
    # Core dependencies
    try:
        import numpy
        results['numpy'] = 'SUCCESS'
    except ImportError:
        results['numpy'] = 'MISSING'
    
    try:
        import jax
        results['jax'] = 'SUCCESS'
    except ImportError:
        results['jax'] = 'MISSING'
    
    try:
        import matplotlib
        results['matplotlib'] = 'SUCCESS'
    except ImportError:
        results['matplotlib'] = 'MISSING'
    
    try:
        import pandas
        results['pandas'] = 'SUCCESS'
    except ImportError:
        results['pandas'] = 'MISSING'
    
    try:
        import scipy
        results['scipy'] = 'SUCCESS'
    except ImportError:
        results['scipy'] = 'MISSING'
    
    # Optional dependencies
    try:
        import pytest
        results['pytest'] = 'SUCCESS'
    except ImportError:
        results['pytest'] = 'MISSING (optional)'
    
    try:
        import coverage
        results['coverage'] = 'SUCCESS'
    except ImportError:
        results['coverage'] = 'MISSING (optional)'
    
    return results

def run_system_health_check():
    """Run system health check."""
    print("Running system health check...")
    
    results = {}
    
    try:
        from phomem.utils.monitoring import run_health_check
        health_results = run_health_check()
        
        for check_name, check_result in health_results.items():
            status_map = {
                'healthy': 'SUCCESS',
                'warning': 'WARNING',
                'critical': 'FAILED'
            }
            results[f'health_{check_name}'] = status_map.get(check_result.status, 'UNKNOWN')
        
    except Exception as e:
        results['health_check'] = f'FAILED: {e}'
    
    return results

def print_results(category, results):
    """Print test results."""
    print(f"\n{category}:")
    print("-" * 40)
    
    success_count = 0
    total_count = 0
    
    for test_name, result in results.items():
        total_count += 1
        if result == 'SUCCESS' or result.startswith('SUCCESS'):
            print(f"  [PASS] {test_name}")
            success_count += 1
        elif 'WARNING' in result:
            print(f"  [WARN] {test_name}: {result}")
        elif 'MISSING (optional)' in result:
            print(f"  [SKIP] {test_name}: {result}")
        else:
            print(f"  [FAIL] {test_name}: {result}")
    
    print(f"\nSummary: {success_count}/{total_count} tests passed ({success_count/total_count*100:.1f}%)")
    return success_count, total_count

def main():
    """Main test runner."""
    print("=" * 60)
    print(" PhoMem-CoSim Quality Gates")
    print("=" * 60)
    print(f"Project Root: {project_root}")
    print(f"Python: {sys.version}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all test categories
    all_results = {}
    total_success = 0
    total_tests = 0
    
    # Basic imports
    import_results = test_basic_imports()
    success, tests = print_results("Import Tests", import_results)
    all_results['imports'] = import_results
    total_success += success
    total_tests += tests
    
    # Basic functionality
    func_results = test_basic_functionality()
    success, tests = print_results("Functionality Tests", func_results)
    all_results['functionality'] = func_results
    total_success += success
    total_tests += tests
    
    # Security features
    security_results = test_security_features()
    success, tests = print_results("Security Tests", security_results)
    all_results['security'] = security_results
    total_success += success
    total_tests += tests
    
    # Dependencies
    dep_results = check_dependencies()
    success, tests = print_results("Dependency Check", dep_results)
    all_results['dependencies'] = dep_results
    # Don't count optional dependencies as failures
    
    # System health
    health_results = run_system_health_check()
    success, tests = print_results("System Health", health_results)
    all_results['health'] = health_results
    total_success += success
    total_tests += tests
    
    # Overall results
    print("\n" + "=" * 60)
    print(" OVERALL RESULTS")
    print("=" * 60)
    
    overall_pass_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {total_success}")
    print(f"Pass Rate: {overall_pass_rate:.1f}%")
    
    if overall_pass_rate >= 90:
        print("Status: EXCELLENT - System ready for production!")
        exit_code = 0
    elif overall_pass_rate >= 70:
        print("Status: GOOD - System functional with minor issues")
        exit_code = 0
    elif overall_pass_rate >= 50:
        print("Status: NEEDS WORK - System has significant issues")
        exit_code = 1
    else:
        print("Status: CRITICAL - System has major issues")
        exit_code = 2
    
    # Save results to file
    report = {
        'timestamp': time.time(),
        'project_root': str(project_root),
        'python_version': sys.version,
        'overall_pass_rate': overall_pass_rate,
        'total_tests': total_tests,
        'successful_tests': total_success,
        'results_by_category': all_results
    }
    
    report_file = project_root / 'quality_gates_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nReport saved to: {report_file}")
    print(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return exit_code

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)