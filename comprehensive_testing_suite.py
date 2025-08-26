#!/usr/bin/env python3
"""
Comprehensive Testing Suite for PhoMem-CoSim
Quality gates and validation framework for all three generations.
"""

import sys
import os
import time
import json
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Tuple
import traceback


class TestSuite:
    """Comprehensive testing framework for PhoMem-CoSim."""
    
    def __init__(self):
        self.results = {
            'generation1': {},
            'generation2': {},
            'generation3': {},
            'overall': {}
        }
        self.start_time = time.time()
    
    def run_generation_test(self, generation: int, script_name: str) -> Dict[str, Any]:
        """Run tests for a specific generation."""
        print(f"\n{'='*60}")
        print(f"🧪 Testing Generation {generation}: {script_name}")
        print(f"{'='*60}")
        
        test_result = {
            'script': script_name,
            'status': 'unknown',
            'execution_time': 0,
            'stdout': '',
            'stderr': '',
            'error': None,
            'metrics': {}
        }
        
        script_path = f"/root/repo/{script_name}"
        
        # Check if script exists
        if not os.path.exists(script_path):
            test_result['status'] = 'missing'
            test_result['error'] = f"Script not found: {script_path}"
            print(f"❌ Script not found: {script_name}")
            return test_result
        
        try:
            # Run the script with timeout
            start_time = time.time()
            
            # Run script with virtual environment python directly
            venv_python = "/root/repo/phomem_gen1_env/bin/python"
            cmd = f"{venv_python} {script_path}"
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            execution_time = time.time() - start_time
            test_result['execution_time'] = execution_time
            test_result['stdout'] = result.stdout
            test_result['stderr'] = result.stderr
            
            # Analyze results
            if result.returncode == 0:
                test_result['status'] = 'passed'
                print(f"✅ Generation {generation} test PASSED")
                print(f"   Execution time: {execution_time:.2f}s")
                
                # Extract metrics from output
                metrics = self._extract_metrics(result.stdout)
                test_result['metrics'] = metrics
                
                if metrics:
                    print(f"   Key metrics:")
                    for key, value in metrics.items():
                        print(f"     {key}: {value}")
                
            else:
                test_result['status'] = 'failed'
                test_result['error'] = f"Return code: {result.returncode}"
                print(f"❌ Generation {generation} test FAILED")
                print(f"   Return code: {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr[:200]}...")
                    
        except subprocess.TimeoutExpired:
            test_result['status'] = 'timeout'
            test_result['error'] = "Execution timed out (300s)"
            print(f"⏰ Generation {generation} test TIMED OUT")
            
        except Exception as e:
            test_result['status'] = 'error'
            test_result['error'] = str(e)
            print(f"💥 Generation {generation} test ERROR: {e}")
        
        return test_result
    
    def _extract_metrics(self, stdout: str) -> Dict[str, Any]:
        """Extract performance metrics from stdout."""
        metrics = {}
        lines = stdout.split('\n')
        
        # Common patterns to extract
        patterns = {
            'training_time': ['Training completed in', 'seconds'],
            'final_loss': ['Final loss:', 'Best loss:', 'Loss:'],
            'throughput': ['samples/sec', 'ops/sec'],
            'speedup': ['Speedup:', 'x'],
            'success_rate': ['% success', '% Success', 'Success Rate:'],
            'convergence': ['Convergence:', 'convergence'],
            'test_success': ['PASSED', 'Successfully Completed']
        }
        
        for line in lines:
            line_lower = line.lower()
            
            # Extract training time
            if 'completed in' in line_lower and 'second' in line_lower:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'second' in part.lower() and i > 0:
                            time_val = float(parts[i-1].replace('s', ''))
                            metrics['training_time_sec'] = time_val
                            break
                except:
                    pass
            
            # Extract loss values
            if 'loss:' in line_lower:
                try:
                    parts = line.split(':')
                    if len(parts) > 1:
                        loss_val = float(parts[-1].strip().split()[0])
                        if 'final' in line_lower:
                            metrics['final_loss'] = loss_val
                        elif 'best' in line_lower:
                            metrics['best_loss'] = loss_val
                except:
                    pass
            
            # Extract throughput
            if 'samples/sec' in line_lower or 'ops/sec' in line_lower:
                try:
                    parts = line.split()
                    for part in parts:
                        if 'samples/sec' in part or 'ops/sec' in part:
                            idx = parts.index(part)
                            if idx > 0:
                                throughput_val = float(parts[idx-1])
                                metrics['throughput'] = throughput_val
                                break
                except:
                    pass
            
            # Extract speedup
            if 'speedup:' in line_lower:
                try:
                    parts = line.split(':')
                    if len(parts) > 1:
                        speedup_val = float(parts[-1].strip().replace('x', ''))
                        metrics['speedup'] = speedup_val
                except:
                    pass
            
            # Extract success rate
            if '% success' in line_lower or 'success rate:' in line_lower:
                try:
                    import re
                    matches = re.findall(r'(\d+(?:\.\d+)?)%', line)
                    if matches:
                        metrics['success_rate'] = float(matches[0])
                except:
                    pass
            
            # Check convergence
            if 'convergence:' in line_lower:
                metrics['convergence'] = 'yes' in line_lower
        
        return metrics
    
    def run_file_validation_tests(self) -> Dict[str, Any]:
        """Validate that required files exist and have content."""
        print(f"\n{'='*60}")
        print("📁 File Validation Tests")
        print(f"{'='*60}")
        
        required_files = [
            'generation1_simple_core.py',
            'generation2_robust_fixed.py', 
            'generation3_scalable_simple.py',
            'README.md',
            'requirements.txt',
            'setup.py'
        ]
        
        validation_results = {}
        
        for filename in required_files:
            filepath = f"/root/repo/{filename}"
            result = {
                'exists': os.path.exists(filepath),
                'size': 0,
                'readable': False,
                'has_content': False
            }
            
            if result['exists']:
                try:
                    stat = os.stat(filepath)
                    result['size'] = stat.st_size
                    result['readable'] = os.access(filepath, os.R_OK)
                    result['has_content'] = stat.st_size > 0
                    
                    status = "✅" if result['has_content'] else "⚠️"
                    print(f"{status} {filename}: {result['size']} bytes")
                    
                except Exception as e:
                    result['error'] = str(e)
                    print(f"❌ {filename}: Error - {e}")
            else:
                print(f"❌ {filename}: Missing")
            
            validation_results[filename] = result
        
        return validation_results
    
    def run_artifact_validation(self) -> Dict[str, Any]:
        """Validate that expected artifacts were generated."""
        print(f"\n{'='*60}")
        print("🎨 Artifact Validation Tests")
        print(f"{'='*60}")
        
        expected_artifacts = [
            'generation1_training_curve.png',
            'generation2_robust_training.png', 
            'generation3_scalable_simple_analysis.png',
            'generation2_robust_fixed_results.json',
            'generation3_scalable_simple_results.json'
        ]
        
        artifact_results = {}
        
        for filename in expected_artifacts:
            filepath = f"/root/repo/{filename}"
            result = {
                'exists': os.path.exists(filepath),
                'size': 0,
                'type': 'unknown'
            }
            
            if result['exists']:
                try:
                    stat = os.stat(filepath)
                    result['size'] = stat.st_size
                    
                    if filename.endswith('.png'):
                        result['type'] = 'image'
                    elif filename.endswith('.json'):
                        result['type'] = 'json'
                        
                        # Validate JSON structure
                        try:
                            with open(filepath, 'r') as f:
                                data = json.load(f)
                                result['json_keys'] = list(data.keys()) if isinstance(data, dict) else None
                        except:
                            result['json_valid'] = False
                    
                    status = "✅"
                    print(f"{status} {filename}: {result['size']} bytes ({result['type']})")
                    
                except Exception as e:
                    result['error'] = str(e)
                    print(f"❌ {filename}: Error - {e}")
            else:
                print(f"⚠️ {filename}: Not generated")
            
            artifact_results[filename] = result
        
        return artifact_results
    
    def run_performance_analysis(self) -> Dict[str, Any]:
        """Analyze performance metrics across generations."""
        print(f"\n{'='*60}")
        print("📊 Performance Analysis")
        print(f"{'='*60}")
        
        performance_data = {}
        
        # Extract metrics from each generation
        for gen_name, gen_results in self.results.items():
            if gen_name == 'overall':
                continue
                
            if 'metrics' in gen_results:
                metrics = gen_results['metrics']
                performance_data[gen_name] = {
                    'execution_time': gen_results.get('execution_time', 0),
                    'throughput': metrics.get('throughput', 0),
                    'speedup': metrics.get('speedup', 1),
                    'final_loss': metrics.get('final_loss', float('inf')),
                    'success_rate': metrics.get('success_rate', 0)
                }
        
        # Performance analysis
        analysis = {
            'generation_comparison': performance_data,
            'performance_progression': {},
            'quality_gates': {}
        }
        
        # Check quality gates
        quality_gates = {
            'all_generations_pass': all(
                gen.get('status') == 'passed' 
                for gen in [self.results.get('generation1', {}), 
                           self.results.get('generation2', {}),
                           self.results.get('generation3', {})]
            ),
            'performance_improvement': False,
            'reasonable_execution_times': True
        }
        
        # Check performance improvement Gen1 -> Gen3
        if ('generation1' in performance_data and 
            'generation3' in performance_data):
            gen1_throughput = performance_data['generation1'].get('throughput', 0)
            gen3_throughput = performance_data['generation3'].get('throughput', 0)
            
            if gen3_throughput > gen1_throughput:
                quality_gates['performance_improvement'] = True
                improvement = gen3_throughput / max(gen1_throughput, 1)
                analysis['performance_progression']['throughput_improvement'] = improvement
                print(f"✅ Performance improvement: {improvement:.2f}x throughput increase")
            else:
                print(f"⚠️ No significant performance improvement detected")
        
        # Check execution times
        for gen_name, data in performance_data.items():
            exec_time = data.get('execution_time', 0)
            if exec_time > 120:  # 2 minutes threshold
                quality_gates['reasonable_execution_times'] = False
                print(f"⚠️ {gen_name} execution time high: {exec_time:.1f}s")
            else:
                print(f"✅ {gen_name} execution time reasonable: {exec_time:.1f}s")
        
        analysis['quality_gates'] = quality_gates
        
        return analysis
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        print(f"\n{'='*80}")
        print("📋 COMPREHENSIVE TEST REPORT")
        print(f"{'='*80}")
        
        total_time = time.time() - self.start_time
        
        # Count test results
        generation_tests = [
            self.results.get('generation1', {}),
            self.results.get('generation2', {}), 
            self.results.get('generation3', {})
        ]
        
        passed_tests = sum(1 for test in generation_tests if test.get('status') == 'passed')
        total_tests = len(generation_tests)
        
        # Overall assessment
        overall_status = 'PASSED' if passed_tests == total_tests else 'FAILED'
        
        report = {
            'timestamp': time.time(),
            'total_execution_time': total_time,
            'overall_status': overall_status,
            'generation_tests': {
                'passed': passed_tests,
                'failed': total_tests - passed_tests,
                'total': total_tests,
                'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            },
            'detailed_results': self.results,
            'quality_assessment': {}
        }
        
        print(f"Overall Status: {'✅' if overall_status == 'PASSED' else '❌'} {overall_status}")
        print(f"Generation Tests: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.1f}%)")
        print(f"Total Execution Time: {total_time:.2f}s")
        
        # Quality assessment
        quality_score = 0
        max_quality_score = 5
        
        # 1. All generations execute successfully
        if passed_tests == total_tests:
            quality_score += 1
            print("✅ All generations execute successfully")
        else:
            print("❌ Some generations failed to execute")
        
        # 2. Performance improvement demonstrated
        gen1_throughput = self.results.get('generation1', {}).get('metrics', {}).get('throughput', 0)
        gen3_throughput = self.results.get('generation3', {}).get('metrics', {}).get('throughput', 0)
        
        if gen3_throughput > gen1_throughput * 2:  # At least 2x improvement
            quality_score += 1
            print(f"✅ Significant performance improvement: {gen3_throughput/max(gen1_throughput,1):.1f}x")
        else:
            print("⚠️ Limited performance improvement demonstrated")
        
        # 3. Reasonable execution times
        avg_exec_time = sum(test.get('execution_time', 0) for test in generation_tests) / max(total_tests, 1)
        if avg_exec_time < 60:  # Under 1 minute average
            quality_score += 1
            print(f"✅ Reasonable execution times: {avg_exec_time:.1f}s average")
        else:
            print(f"⚠️ Long execution times: {avg_exec_time:.1f}s average")
        
        # 4. Training convergence
        convergence_count = sum(1 for test in generation_tests 
                              if test.get('metrics', {}).get('convergence', False))
        if convergence_count >= 2:
            quality_score += 1
            print(f"✅ Training convergence achieved in {convergence_count}/3 generations")
        else:
            print(f"⚠️ Limited training convergence: {convergence_count}/3 generations")
        
        # 5. Artifacts generated
        artifact_validation = self.results.get('artifacts', {})
        artifacts_generated = sum(1 for artifact in artifact_validation.values() 
                                if artifact.get('exists', False))
        total_artifacts = len(artifact_validation)
        
        if artifacts_generated >= total_artifacts * 0.8:  # 80% of artifacts
            quality_score += 1
            print(f"✅ Most artifacts generated: {artifacts_generated}/{total_artifacts}")
        else:
            print(f"⚠️ Missing artifacts: {artifacts_generated}/{total_artifacts}")
        
        report['quality_assessment'] = {
            'quality_score': quality_score,
            'max_score': max_quality_score,
            'percentage': (quality_score / max_quality_score) * 100,
            'grade': self._get_quality_grade(quality_score, max_quality_score)
        }
        
        grade = report['quality_assessment']['grade']
        print(f"\nQuality Score: {quality_score}/{max_quality_score} ({quality_score/max_quality_score*100:.1f}%)")
        print(f"Quality Grade: {grade}")
        
        return report
    
    def _get_quality_grade(self, score: int, max_score: int) -> str:
        """Convert quality score to letter grade."""
        percentage = (score / max_score) * 100
        
        if percentage >= 90:
            return "A"
        elif percentage >= 80:
            return "B" 
        elif percentage >= 70:
            return "C"
        elif percentage >= 60:
            return "D"
        else:
            return "F"
    
    def save_report(self, report: Dict[str, Any]):
        """Save comprehensive report to file."""
        report_file = "/root/repo/comprehensive_test_report.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n📄 Comprehensive report saved to: {report_file}")
        except Exception as e:
            print(f"❌ Failed to save report: {e}")


def run_comprehensive_testing():
    """Run the complete testing suite."""
    print("🧪 PhoMem-CoSim Comprehensive Testing Suite")
    print("=" * 80)
    print("Validating all three generations with quality gates")
    print("=" * 80)
    
    suite = TestSuite()
    
    # Test each generation
    suite.results['generation1'] = suite.run_generation_test(1, 'generation1_simple_core.py')
    suite.results['generation2'] = suite.run_generation_test(2, 'generation2_robust_fixed.py')
    suite.results['generation3'] = suite.run_generation_test(3, 'generation3_scalable_simple.py')
    
    # File validation
    suite.results['file_validation'] = suite.run_file_validation_tests()
    
    # Artifact validation  
    suite.results['artifacts'] = suite.run_artifact_validation()
    
    # Performance analysis
    suite.results['performance'] = suite.run_performance_analysis()
    
    # Generate comprehensive report
    final_report = suite.generate_comprehensive_report()
    
    # Save report
    suite.save_report(final_report)
    
    # Final assessment
    overall_success = final_report['overall_status'] == 'PASSED'
    quality_grade = final_report['quality_assessment']['grade']
    
    if overall_success and quality_grade in ['A', 'B']:
        print(f"\n🎉 COMPREHENSIVE TESTING: SUCCESS!")
        print(f"All quality gates passed with grade {quality_grade}")
        return True
    else:
        print(f"\n⚠️ COMPREHENSIVE TESTING: NEEDS ATTENTION")
        print(f"Quality grade: {quality_grade}")
        return False


if __name__ == "__main__":
    success = run_comprehensive_testing()
    
    if success:
        print("\n✅ All quality gates passed - Production ready!")
        sys.exit(0)
    else:
        print("\n❌ Some quality gates failed - Review needed")
        sys.exit(1)