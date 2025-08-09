#!/usr/bin/env python3
"""
Code structure and quality tests for PhoMem-CoSim (no external dependencies).
"""

import sys
import os
import time
import json
import ast
import re
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_file_structure():
    """Test project file structure."""
    print("Testing file structure...")
    
    results = {}
    
    # Check for essential files
    essential_files = [
        'README.md',
        'setup.py', 
        'requirements.txt',
        'phomem/__init__.py',
        'phomem/config.py',
        'phomem/utils/__init__.py',
        'phomem/utils/data.py',
        'phomem/utils/plotting.py',
        'phomem/utils/security.py',
        'phomem/utils/exceptions.py'
    ]
    
    for file_path in essential_files:
        full_path = project_root / file_path
        if full_path.exists():
            results[file_path] = 'SUCCESS'
        else:
            results[file_path] = 'MISSING'
    
    # Check directory structure
    essential_dirs = [
        'phomem',
        'phomem/utils',
        'phomem/neural',
        'phomem/photonics',
        'phomem/memristors',
        'phomem/simulator',
        'deployment',
        'examples'
    ]
    
    for dir_path in essential_dirs:
        full_path = project_root / dir_path
        if full_path.exists() and full_path.is_dir():
            results[f'dir_{dir_path}'] = 'SUCCESS'
        else:
            results[f'dir_{dir_path}'] = 'MISSING'
    
    return results

def test_python_syntax():
    """Test Python syntax of all .py files."""
    print("Testing Python syntax...")
    
    results = {}
    
    # Find all Python files
    python_files = list(project_root.glob('**/*.py'))
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST to check syntax
            ast.parse(content)
            results[str(py_file.relative_to(project_root))] = 'SUCCESS'
            
        except SyntaxError as e:
            results[str(py_file.relative_to(project_root))] = f'SYNTAX_ERROR: {e}'
        except UnicodeDecodeError:
            results[str(py_file.relative_to(project_root))] = 'ENCODING_ERROR'
        except Exception as e:
            results[str(py_file.relative_to(project_root))] = f'ERROR: {e}'
    
    return results

def test_import_structure():
    """Test import structure without actually importing."""
    print("Testing import structure...")
    
    results = {}
    
    # Test __init__.py files
    init_files = list(project_root.glob('**/__init__.py'))
    
    for init_file in init_files:
        try:
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for __all__ exports
            if '__all__' in content:
                results[f'exports_{init_file.parent.name}'] = 'SUCCESS'
            else:
                results[f'exports_{init_file.parent.name}'] = 'WARNING: No __all__ defined'
            
            # Check for version info in main __init__.py
            if init_file.name == '__init__.py' and 'phomem' in str(init_file):
                if '__version__' in content:
                    results['version_info'] = 'SUCCESS'
                else:
                    results['version_info'] = 'WARNING: No version info'
            
        except Exception as e:
            results[f'init_{init_file.parent.name}'] = f'ERROR: {e}'
    
    return results

def test_docstrings():
    """Test docstring coverage."""
    print("Testing docstrings...")
    
    results = {}
    
    # Find Python files in main package
    py_files = list((project_root / 'phomem').glob('**/*.py'))
    
    total_functions = 0
    documented_functions = 0
    total_classes = 0
    documented_classes = 0
    
    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to find functions and classes
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    total_functions += 1
                    if ast.get_docstring(node):
                        documented_functions += 1
                
                elif isinstance(node, ast.ClassDef):
                    total_classes += 1
                    if ast.get_docstring(node):
                        documented_classes += 1
            
        except Exception as e:
            results[f'docstring_error_{py_file.name}'] = f'ERROR: {e}'
    
    # Calculate coverage
    if total_functions > 0:
        func_coverage = (documented_functions / total_functions) * 100
        results['function_docstring_coverage'] = f'{func_coverage:.1f}% ({documented_functions}/{total_functions})'
    
    if total_classes > 0:
        class_coverage = (documented_classes / total_classes) * 100
        results['class_docstring_coverage'] = f'{class_coverage:.1f}% ({documented_classes}/{total_classes})'
    
    return results

def test_code_quality():
    """Test basic code quality metrics."""
    print("Testing code quality...")
    
    results = {}
    
    # Find Python files
    py_files = list((project_root / 'phomem').glob('**/*.py'))
    
    total_lines = 0
    total_files = 0
    long_functions = 0
    complex_functions = 0
    
    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            total_files += 1
            file_lines = len([line for line in lines if line.strip()])  # Non-empty lines
            total_lines += file_lines
            
            # Parse for function complexity
            content = ''.join(lines)
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Count lines in function
                    func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                    if func_lines > 50:  # Functions longer than 50 lines
                        long_functions += 1
                    
                    # Count complexity (simplified)
                    complexity = 1  # Base complexity
                    for child in ast.walk(node):
                        if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                            complexity += 1
                    
                    if complexity > 10:  # High complexity
                        complex_functions += 1
            
        except Exception as e:
            results[f'quality_error_{py_file.name}'] = f'ERROR: {e}'
    
    # Calculate metrics
    if total_files > 0:
        avg_file_length = total_lines / total_files
        results['avg_file_length'] = f'{avg_file_length:.0f} lines'
        results['total_lines_of_code'] = f'{total_lines} lines'
        results['total_files'] = f'{total_files} files'
    
    if long_functions == 0:
        results['long_functions'] = 'SUCCESS: No functions > 50 lines'
    else:
        results['long_functions'] = f'WARNING: {long_functions} functions > 50 lines'
    
    if complex_functions == 0:
        results['complex_functions'] = 'SUCCESS: No highly complex functions'
    else:
        results['complex_functions'] = f'WARNING: {complex_functions} highly complex functions'
    
    return results

def test_security_patterns():
    """Test for basic security anti-patterns."""
    print("Testing security patterns...")
    
    results = {}
    
    # Security patterns to check
    dangerous_patterns = [
        (r'exec\s*\(', 'exec() usage'),
        (r'eval\s*\(', 'eval() usage'),
        (r'__import__', 'dynamic imports'),
        (r'subprocess\.call', 'subprocess.call usage'),
        (r'os\.system', 'os.system usage'),
        (r'shell\s*=\s*True', 'shell=True in subprocess'),
        (r'pickle\.loads?\(', 'pickle usage (potential security risk)'),
    ]
    
    # Find Python files
    py_files = list(project_root.glob('**/*.py'))
    
    total_issues = 0
    files_with_issues = 0
    
    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_issues = 0
            for pattern, description in dangerous_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    file_issues += len(matches)
                    total_issues += len(matches)
            
            if file_issues > 0:
                files_with_issues += 1
                results[f'security_{py_file.name}'] = f'WARNING: {file_issues} potential issues'
            
        except Exception as e:
            results[f'security_error_{py_file.name}'] = f'ERROR: {e}'
    
    # Summary
    if total_issues == 0:
        results['security_summary'] = 'SUCCESS: No obvious security issues found'
    else:
        results['security_summary'] = f'WARNING: {total_issues} potential issues in {files_with_issues} files'
    
    return results

def test_configuration_files():
    """Test configuration files."""
    print("Testing configuration files...")
    
    results = {}
    
    # Test setup.py
    setup_file = project_root / 'setup.py'
    if setup_file.exists():
        try:
            with open(setup_file, 'r') as f:
                content = f.read()
            
            required_fields = ['name', 'version', 'author', 'description']
            missing_fields = []
            
            for field in required_fields:
                if field not in content:
                    missing_fields.append(field)
            
            if not missing_fields:
                results['setup_py'] = 'SUCCESS'
            else:
                results['setup_py'] = f'WARNING: Missing fields {missing_fields}'
        
        except Exception as e:
            results['setup_py'] = f'ERROR: {e}'
    else:
        results['setup_py'] = 'MISSING'
    
    # Test requirements.txt
    req_file = project_root / 'requirements.txt'
    if req_file.exists():
        try:
            with open(req_file, 'r') as f:
                lines = f.readlines()
            
            requirements = [line.strip() for line in lines if line.strip() and not line.startswith('#')]
            
            if len(requirements) > 0:
                results['requirements_txt'] = f'SUCCESS: {len(requirements)} requirements'
            else:
                results['requirements_txt'] = 'WARNING: No requirements specified'
        
        except Exception as e:
            results['requirements_txt'] = f'ERROR: {e}'
    else:
        results['requirements_txt'] = 'MISSING'
    
    # Test README.md
    readme_file = project_root / 'README.md'
    if readme_file.exists():
        try:
            with open(readme_file, 'r') as f:
                content = f.read()
            
            if len(content) > 500:  # Reasonable README size
                results['readme'] = 'SUCCESS'
            else:
                results['readme'] = 'WARNING: README seems short'
        
        except Exception as e:
            results['readme'] = f'ERROR: {e}'
    else:
        results['readme'] = 'MISSING'
    
    return results

def print_results(category, results):
    """Print test results."""
    print(f"\n{category}:")
    print("-" * 50)
    
    success_count = 0
    warning_count = 0
    error_count = 0
    total_count = len(results)
    
    for test_name, result in results.items():
        if result == 'SUCCESS' or result.startswith('SUCCESS'):
            print(f"  [PASS] {test_name}: {result}")
            success_count += 1
        elif 'WARNING' in result:
            print(f"  [WARN] {test_name}: {result}")
            warning_count += 1
        elif 'ERROR' in result:
            print(f"  [FAIL] {test_name}: {result}")
            error_count += 1
        elif result == 'MISSING':
            print(f"  [FAIL] {test_name}: {result}")
            error_count += 1
        else:
            print(f"  [INFO] {test_name}: {result}")
    
    print(f"\nSummary: {success_count} passed, {warning_count} warnings, {error_count} failed ({success_count}/{total_count} success rate: {success_count/total_count*100:.1f}%)")
    return success_count, warning_count, error_count

def main():
    """Main test runner."""
    print("=" * 60)
    print(" PhoMem-CoSim Code Structure & Quality Tests")
    print("=" * 60)
    print(f"Project Root: {project_root}")
    print(f"Python: {sys.version}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all test categories
    all_results = {}
    total_success = 0
    total_warnings = 0
    total_errors = 0
    
    # File structure
    structure_results = test_file_structure()
    success, warnings, errors = print_results("File Structure", structure_results)
    all_results['structure'] = structure_results
    total_success += success
    total_warnings += warnings
    total_errors += errors
    
    # Python syntax
    syntax_results = test_python_syntax()
    success, warnings, errors = print_results("Python Syntax", syntax_results)
    all_results['syntax'] = syntax_results
    total_success += success
    total_warnings += warnings
    total_errors += errors
    
    # Import structure
    import_results = test_import_structure()
    success, warnings, errors = print_results("Import Structure", import_results)
    all_results['imports'] = import_results
    total_success += success
    total_warnings += warnings
    total_errors += errors
    
    # Docstrings
    doc_results = test_docstrings()
    success, warnings, errors = print_results("Documentation", doc_results)
    all_results['documentation'] = doc_results
    total_success += success
    total_warnings += warnings
    total_errors += errors
    
    # Code quality
    quality_results = test_code_quality()
    success, warnings, errors = print_results("Code Quality", quality_results)
    all_results['quality'] = quality_results
    total_success += success
    total_warnings += warnings
    total_errors += errors
    
    # Security patterns
    security_results = test_security_patterns()
    success, warnings, errors = print_results("Security Patterns", security_results)
    all_results['security'] = security_results
    total_success += success
    total_warnings += warnings
    total_errors += errors
    
    # Configuration files
    config_results = test_configuration_files()
    success, warnings, errors = print_results("Configuration Files", config_results)
    all_results['configuration'] = config_results
    total_success += success
    total_warnings += warnings
    total_errors += errors
    
    # Overall results
    print("\n" + "=" * 60)
    print(" OVERALL RESULTS")
    print("=" * 60)
    
    total_tests = total_success + total_warnings + total_errors
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_success}")
    print(f"Warnings: {total_warnings}")
    print(f"Failed: {total_errors}")
    
    if total_tests > 0:
        pass_rate = (total_success / total_tests) * 100
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        if pass_rate >= 90 and total_errors == 0:
            print("Status: EXCELLENT - Code structure is excellent!")
            exit_code = 0
        elif pass_rate >= 75:
            print("Status: GOOD - Code structure is solid with minor issues")
            exit_code = 0
        elif pass_rate >= 50:
            print("Status: NEEDS WORK - Code structure needs improvement")
            exit_code = 1
        else:
            print("Status: CRITICAL - Major code structure issues")
            exit_code = 2
    else:
        print("Status: NO TESTS RUN")
        exit_code = 1
    
    # Save results
    report = {
        'timestamp': time.time(),
        'project_root': str(project_root),
        'python_version': sys.version,
        'total_tests': total_tests,
        'passed': total_success,
        'warnings': total_warnings,
        'failed': total_errors,
        'pass_rate': (total_success / total_tests) * 100 if total_tests > 0 else 0,
        'results_by_category': all_results
    }
    
    report_file = project_root / 'code_structure_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: {report_file}")
    print(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return exit_code

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)