"""
Command-line interface for PhoMem-CoSim operations.
"""

import argparse
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from . import __version__
from .neural.networks import HybridNetwork
from .neural.training import HardwareAwareTrainer
from .simulator.multiphysics import MultiPhysicsSimulator
from .utils.logging import setup_logging
from .utils.performance import ProfileManager
from .utils.validation import ConfigValidator


def setup_cli_logging(level: str = "INFO"):
    """Setup logging for CLI operations."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    setup_logging(
        log_level=numeric_level,
        log_file=Path("phomem_cli.log"),
        console_output=True
    )


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        # Validate configuration
        validator = ConfigValidator()
        validator.validate_config(config)
        
        return config
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)


def simulate_command(args):
    """Execute simulation command."""
    setup_cli_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting PhoMem-CoSim simulation v{__version__}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {args.output}")
    
    # Load configuration
    config = load_config(Path(args.config))
    
    # Setup profiling if requested
    with ProfileManager(enabled=args.profile, output_dir=args.output) as profiler:
        try:
            # Initialize simulator
            simulator = MultiPhysicsSimulator(
                optical_solver=config.get('optical_solver', 'BPM'),
                thermal_solver=config.get('thermal_solver', 'FEM'),
                electrical_solver=config.get('electrical_solver', 'SPICE'),
                coupling=config.get('coupling', 'weak')
            )
            
            # Load or create network
            if 'network_path' in config:
                network = HybridNetwork.load(config['network_path'])
            else:
                network = HybridNetwork.from_config(config['network'])
            
            # Run simulation
            results = simulator.simulate(
                network=network,
                inputs=jnp.array(config['inputs']),
                duration=config.get('duration', 1.0),
                save_fields=config.get('save_fields', False)
            )
            
            # Save results
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save as NumPy arrays for easy loading
            np.savez_compressed(
                output_path / "simulation_results.npz",
                **results
            )
            
            # Save configuration for reproducibility
            with open(output_path / "simulation_config.yaml", 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Simulation completed successfully. Results saved to {output_path}")
            
            # Print summary
            print(f"Simulation Summary:")
            print(f"  Duration: {results.get('simulation_time', 'N/A')} seconds")
            print(f"  Energy: {results.get('total_energy', 'N/A')} J")
            print(f"  Peak Power: {results.get('peak_power', 'N/A')} W")
            print(f"  Results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            sys.exit(1)


def train_command(args):
    """Execute training command."""
    setup_cli_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting PhoMem-CoSim training v{__version__}")
    
    # Load configuration
    config = load_config(Path(args.config))
    
    # Setup profiling if requested
    with ProfileManager(enabled=args.profile, output_dir=args.output) as profiler:
        try:
            # Initialize trainer
            trainer = HardwareAwareTrainer(
                learning_rate=config.get('learning_rate', 1e-3),
                hardware_penalties=config.get('hardware_penalties', {}),
                optimization_config=config.get('optimization', {})
            )
            
            # Load network
            if 'network_path' in config:
                network = HybridNetwork.load(config['network_path'])
            else:
                network = HybridNetwork.from_config(config['network'])
            
            # Load training data
            data_config = config['data']
            if data_config['type'] == 'synthetic':
                # Generate synthetic data for testing
                from .utils.data import generate_synthetic_dataset
                train_data = generate_synthetic_dataset(
                    size=data_config.get('size', 1000),
                    input_dim=network.input_size,
                    output_dim=network.output_size
                )
            else:
                # Load real dataset
                train_data = np.load(data_config['path'])
            
            # Train network
            trained_network, training_history = trainer.train(
                network=network,
                data=train_data,
                epochs=config.get('epochs', 100),
                batch_size=config.get('batch_size', 32),
                validation_split=config.get('validation_split', 0.2)
            )
            
            # Save trained model
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            trained_network.save(output_path / "trained_model.pkl")
            
            # Save training history
            with open(output_path / "training_history.json", 'w') as f:
                json.dump(training_history, f, indent=2)
            
            logger.info(f"Training completed successfully. Model saved to {output_path}")
            
            # Print training summary
            final_loss = training_history['loss'][-1]
            final_accuracy = training_history.get('accuracy', [0])[-1]
            
            print(f"Training Summary:")
            print(f"  Final Loss: {final_loss:.6f}")
            print(f"  Final Accuracy: {final_accuracy:.3f}")
            print(f"  Epochs: {len(training_history['loss'])}")
            print(f"  Model saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            sys.exit(1)


def analyze_command(args):
    """Execute analysis command."""
    setup_cli_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting PhoMem-CoSim analysis v{__version__}")
    
    # Load configuration
    config = load_config(Path(args.config))
    
    try:
        # Import analysis modules
        from .simulator.optimization import PerformanceAnalyzer, VariabilityAnalyzer
        from .utils.plotting import create_analysis_plots
        
        # Initialize analyzers
        perf_analyzer = PerformanceAnalyzer()
        var_analyzer = VariabilityAnalyzer()
        
        # Load network or simulation results
        if args.network:
            network = HybridNetwork.load(args.network)
            subject = network
        elif args.results:
            results = np.load(args.results)
            subject = results
        else:
            raise ValueError("Must specify either --network or --results")
        
        # Run requested analyses
        analysis_results = {}
        
        if 'performance' in config.get('analyses', []):
            logger.info("Running performance analysis...")
            perf_results = perf_analyzer.analyze(subject)
            analysis_results['performance'] = perf_results
        
        if 'variability' in config.get('analyses', []):
            logger.info("Running variability analysis...")
            var_results = var_analyzer.analyze(
                subject,
                n_samples=config.get('variability_samples', 1000)
            )
            analysis_results['variability'] = var_results
        
        # Generate plots and reports
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save analysis results
        with open(output_path / "analysis_results.json", 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Generate plots
        if args.plots:
            logger.info("Generating analysis plots...")
            plot_paths = create_analysis_plots(
                analysis_results,
                output_dir=output_path / "plots"
            )
            logger.info(f"Plots saved to {output_path / 'plots'}")
        
        logger.info(f"Analysis completed successfully. Results saved to {output_path}")
        
        # Print analysis summary
        print(f"Analysis Summary:")
        for analysis_type, results in analysis_results.items():
            print(f"  {analysis_type.capitalize()}:")
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        print(f"    {key}: {value:.3e}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PhoMem-CoSim: Photonic-Memristive Neural Network Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'PhoMem-CoSim {__version__}'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Configuration file (YAML or JSON)'
    )
    common_parser.add_argument(
        '--output', '-o',
        type=str,
        default='./phomem_output',
        help='Output directory (default: ./phomem_output)'
    )
    common_parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    common_parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling'
    )
    
    # Simulate command
    simulate_parser = subparsers.add_parser(
        'simulate',
        parents=[common_parser],
        help='Run photonic-memristive simulation'
    )
    simulate_parser.set_defaults(func=simulate_command)
    
    # Train command  
    train_parser = subparsers.add_parser(
        'train',
        parents=[common_parser],
        help='Train hybrid neural network'
    )
    train_parser.set_defaults(func=train_command)
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        parents=[common_parser],
        help='Analyze network or simulation results'
    )
    analyze_parser.add_argument(
        '--network',
        type=str,
        help='Path to trained network file'
    )
    analyze_parser.add_argument(
        '--results',
        type=str,
        help='Path to simulation results file'
    )
    analyze_parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate analysis plots'
    )
    analyze_parser.set_defaults(func=analyze_command)
    
    # Parse arguments and execute
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Set JAX platform
    if hasattr(args, 'device'):
        jax.config.update('jax_platform_name', args.device)
    
    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()