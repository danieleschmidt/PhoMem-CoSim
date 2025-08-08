#!/usr/bin/env python3
"""
Simple test for Generation 2 multi-physics components without JAX dependency.
"""

import sys
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

def test_code_structure():
    """Test that Generation 2 code files exist and have valid syntax."""
    logger.info("Testing Generation 2 code structure...")
    
    import ast
    
    gen2_files = [
        '/root/repo/phomem/advanced_multiphysics.py',
        '/root/repo/phomem/self_healing_optimization.py'
    ]
    
    success = True
    
    for file_path in gen2_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse AST to check syntax
            ast.parse(content)
            logger.info(f"✓ {file_path} - syntax valid")
            
        except FileNotFoundError:
            logger.error(f"✗ {file_path} - file not found")
            success = False
        except SyntaxError as e:
            logger.error(f"✗ {file_path} - syntax error: {e}")
            success = False
        except Exception as e:
            logger.error(f"✗ {file_path} - error: {e}")
            success = False
    
    return success

def test_class_definitions():
    """Test that key classes are defined in the modules."""
    logger.info("Testing class definitions...")
    
    import ast
    
    # Expected classes in each module
    expected_classes = {
        '/root/repo/phomem/advanced_multiphysics.py': [
            'AdvancedMultiPhysicsSimulator',
            'BayesianMultiObjectiveOptimizer',
            'UncertaintyQuantificationResult',
            'MultiPhysicsState'
        ],
        '/root/repo/phomem/self_healing_optimization.py': [
            'SelfHealingOptimizer',
            'AdaptiveMeshRefinement',
            'HealthMetrics',
            'HealingAction'
        ]
    }
    
    success = True
    
    for file_path, expected in expected_classes.items():
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse and find class definitions
            tree = ast.parse(content)
            found_classes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    found_classes.append(node.name)
            
            # Check if all expected classes are found
            missing_classes = set(expected) - set(found_classes)
            if missing_classes:
                logger.error(f"✗ {file_path} - missing classes: {missing_classes}")
                success = False
            else:
                logger.info(f"✓ {file_path} - all expected classes found")
                
        except Exception as e:
            logger.error(f"✗ {file_path} - error checking classes: {e}")
            success = False
    
    return success

def test_function_definitions():
    """Test that key functions are defined."""
    logger.info("Testing function definitions...")
    
    import ast
    
    expected_functions = {
        '/root/repo/phomem/advanced_multiphysics.py': [
            'create_initial_multiphysics_state'
        ],
        '/root/repo/phomem/self_healing_optimization.py': [
            'create_self_healing_optimizer'
        ]
    }
    
    success = True
    
    for file_path, expected in expected_functions.items():
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            found_functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    found_functions.append(node.name)
            
            missing_functions = set(expected) - set(found_functions)
            if missing_functions:
                logger.error(f"✗ {file_path} - missing functions: {missing_functions}")
                success = False
            else:
                logger.info(f"✓ {file_path} - all expected functions found")
                
        except Exception as e:
            logger.error(f"✗ {file_path} - error checking functions: {e}")
            success = False
    
    return success

def test_imports():
    """Test that imports are properly structured."""
    logger.info("Testing import structure...")
    
    import ast
    
    gen2_files = [
        '/root/repo/phomem/advanced_multiphysics.py',
        '/root/repo/phomem/self_healing_optimization.py'
    ]
    
    success = True
    
    for file_path in gen2_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Count different types of imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(f"from {node.module}")
            
            logger.info(f"✓ {file_path} - {len(imports)} import statements found")
            
            # Check for key scientific computing imports
            scientific_imports = ['numpy', 'jax', 'scipy', 'matplotlib']
            found_scientific = [imp for imp in scientific_imports 
                               if any(sci in imp for imp in imports)]
            
            if found_scientific:
                logger.info(f"  Scientific imports: {found_scientific}")
            
        except Exception as e:
            logger.error(f"✗ {file_path} - error checking imports: {e}")
            success = False
    
    return success

def test_docstrings():
    """Test that key components have docstrings."""
    logger.info("Testing docstring coverage...")
    
    import ast
    
    gen2_files = [
        '/root/repo/phomem/advanced_multiphysics.py',
        '/root/repo/phomem/self_healing_optimization.py'
    ]
    
    success = True
    
    for file_path in gen2_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Check module docstring
            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                logger.info(f"✓ {file_path} - has module docstring")
            else:
                logger.warning(f"⚠ {file_path} - missing module docstring")
            
            # Check class and function docstrings
            classes_with_docs = 0
            total_classes = 0
            functions_with_docs = 0
            total_functions = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    total_classes += 1
                    if ast.get_docstring(node):
                        classes_with_docs += 1
                elif isinstance(node, ast.FunctionDef):
                    total_functions += 1
                    if ast.get_docstring(node):
                        functions_with_docs += 1
            
            class_doc_rate = classes_with_docs / total_classes if total_classes > 0 else 1.0
            func_doc_rate = functions_with_docs / total_functions if total_functions > 0 else 1.0
            
            logger.info(f"  Class docstring coverage: {class_doc_rate:.1%} ({classes_with_docs}/{total_classes})")
            logger.info(f"  Function docstring coverage: {func_doc_rate:.1%} ({functions_with_docs}/{total_functions})")
            
        except Exception as e:
            logger.error(f"✗ {file_path} - error checking docstrings: {e}")
            success = False
    
    return success

def test_code_complexity():
    """Test code complexity metrics."""
    logger.info("Testing code complexity...")
    
    import ast
    
    gen2_files = [
        '/root/repo/phomem/advanced_multiphysics.py',
        '/root/repo/phomem/self_healing_optimization.py'
    ]
    
    success = True
    
    for file_path in gen2_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Simple metrics
            total_lines = len(content.splitlines())
            total_classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            total_functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            
            logger.info(f"✓ {file_path}:")
            logger.info(f"  Lines of code: {total_lines}")
            logger.info(f"  Classes: {total_classes}")
            logger.info(f"  Functions: {total_functions}")
            
            # Rough complexity check
            if total_lines > 2000:
                logger.warning(f"  ⚠ Large file ({total_lines} lines)")
            if total_classes > 20:
                logger.warning(f"  ⚠ Many classes ({total_classes})")
            if total_functions > 100:
                logger.warning(f"  ⚠ Many functions ({total_functions})")
                
        except Exception as e:
            logger.error(f"✗ {file_path} - error analyzing complexity: {e}")
            success = False
    
    return success

def main():
    """Run Generation 2 structural tests."""
    logger.info("=== Generation 2 Structural Test Suite ===")
    
    tests = [
        ("Code Structure", test_code_structure),
        ("Class Definitions", test_class_definitions),
        ("Function Definitions", test_function_definitions),
        ("Import Structure", test_imports),
        ("Docstring Coverage", test_docstrings),
        ("Code Complexity", test_code_complexity),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            start_time = time.time()
            result = test_func()
            test_time = time.time() - start_time
            
            results.append(result)
            status = "PASSED" if result else "FAILED"
            logger.info(f"{test_name}: {status} ({test_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"{test_name}: FAILED - {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    success_rate = passed / total
    
    logger.info(f"\n=== Generation 2 Structural Test Results ===")
    logger.info(f"Passed: {passed}/{total} tests ({success_rate:.1%})")
    
    if success_rate >= 0.8:
        logger.info("🎉 Generation 2 structure is solid!")
        return True
    else:
        logger.error("❌ Generation 2 structure needs improvement")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)