#!/usr/bin/env python3
"""
Comprehensive Research Validation Study

This script runs the complete validation study for all novel optimization algorithms
and generates publication-ready results and documentation.
"""

import logging
import sys
import traceback
from pathlib import Path

# Add the phomem module to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from phomem.comprehensive_benchmarking import ComprehensiveBenchmarkSuite
from phomem.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(level=logging.INFO)
logger = get_logger(__name__)


def main():
    """Run comprehensive research validation study."""
    
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE RESEARCH VALIDATION STUDY")
    logger.info("Novel Optimization Algorithms for Photonic-Memristive Systems")
    logger.info("=" * 80)
    
    try:
        # Initialize benchmark suite
        logger.info("Initializing Comprehensive Benchmark Suite...")
        benchmark_suite = ComprehensiveBenchmarkSuite(
            output_directory="./research_validation_results",
            enable_profiling=True,
            parallel_workers=4  # Adjust based on system capabilities
        )
        
        # Create main benchmark configuration
        logger.info("Creating comprehensive benchmark configuration...")
        main_benchmark = benchmark_suite.create_comprehensive_benchmark(
            benchmark_name="Comprehensive_Novel_Optimization_Algorithms_Study_2025",
            algorithm_categories=['quantum', 'self_healing', 'pinas'],
            test_function_categories=['classical', 'photonic_specific'],
            num_trials=5  # Balanced between thoroughness and computational time
        )
        
        logger.info(f"Benchmark configuration created:")
        logger.info(f"  - Algorithms: {len(main_benchmark.algorithms)}")
        logger.info(f"  - Test functions: {len(main_benchmark.test_functions)}")
        logger.info(f"  - Trials per combination: {main_benchmark.num_trials}")
        logger.info(f"  - Total experiments: {len(main_benchmark.algorithms) * len(main_benchmark.test_functions) * main_benchmark.num_trials}")
        
        # Run comprehensive benchmark
        logger.info("Starting comprehensive benchmark execution...")
        logger.info("This may take significant time depending on system capabilities...")
        
        main_result = benchmark_suite.run_comprehensive_benchmark(main_benchmark)
        
        logger.info("Comprehensive benchmark completed successfully!")
        logger.info(f"Execution time: {main_result.execution_time:.2f} seconds")
        logger.info(f"Best overall algorithm: {main_result.publication_summary['overall_best_algorithm']}")
        
        # Generate publication-quality plots
        logger.info("Generating publication-quality visualizations...")
        benchmark_suite.generate_publication_plots(main_result)
        
        # Print key findings
        logger.info("\n" + "=" * 60)
        logger.info("KEY RESEARCH FINDINGS")
        logger.info("=" * 60)
        
        pub_summary = main_result.publication_summary
        
        logger.info(f"Total algorithms tested: {pub_summary['total_algorithms_tested']}")
        logger.info(f"Total test functions: {pub_summary['total_test_functions']}")
        logger.info(f"Significant improvements rate: {pub_summary['significant_improvements_rate']:.1%}")
        
        logger.info("\nAlgorithm Categories Performance:")
        for category, performance in pub_summary['category_performance'].items():
            logger.info(f"  {category}: Best rank {performance['best_rank']}, Avg rank {performance['average_rank']:.1f}")
        
        logger.info("\nKey Findings:")
        for finding in pub_summary['key_findings']:
            logger.info(f"  • {finding}")
        
        # Performance rankings
        logger.info("\nOverall Performance Rankings:")
        for algo, rank in sorted(main_result.performance_rankings.items(), key=lambda x: x[1]):
            logger.info(f"  {rank}. {algo}")
        
        # Statistical significance summary
        logger.info(f"\nStatistical Analysis:")
        logger.info(f"  Significance level: {main_result.statistical_analysis['significance_level']}")
        
        # Computational complexity summary
        if 'efficiency_rankings' in main_result.computational_complexity:
            logger.info("\nComputational Efficiency Rankings:")
            for algo, rank in sorted(main_result.computational_complexity['efficiency_rankings'].items(), key=lambda x: x[1]):
                logger.info(f"  {rank}. {algo}")
        
        # Robustness rankings
        if 'robustness_rankings' in main_result.robustness_analysis:
            logger.info("\nRobustness Rankings:")
            for algo, rank in sorted(main_result.robustness_analysis['robustness_rankings'].items(), key=lambda x: x[1]):
                logger.info(f"  {rank}. {algo}")
        
        logger.info("\n" + "=" * 60)
        logger.info("RESEARCH VALIDATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Results saved in: ./research_validation_results/")
        logger.info("Publication-ready plots and data are available for academic submission.")
        
        return main_result
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("RESEARCH VALIDATION FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    result = main()
    if result is not None:
        sys.exit(0)
    else:
        sys.exit(1)