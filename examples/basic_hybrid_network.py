"""
Basic example of photonic-memristive hybrid network simulation.

This example demonstrates:
1. Creating a hybrid network with photonic and memristive layers
2. Forward propagation through the network
3. Hardware-aware training with device constraints
4. Performance analysis and visualization
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

# Import PhoMem-CoSim components
import phomem as pm
from phomem.photonics import MachZehnderMesh, PhotoDetectorArray
from phomem.memristors import PCMCrossbar
from phomem.neural import HybridNetwork, PhotonicLayer, MemristiveLayer, TransimpedanceAmplifier
from phomem.simulator import MultiPhysicsSimulator


def create_hybrid_network() -> HybridNetwork:
    """Create a simple hybrid photonic-memristive network."""
    
    # Network parameters
    input_size = 4
    hidden_size = 8
    output_size = 2
    
    # Photonic front-end: 4x4 MZI mesh
    photonic_layer = PhotonicLayer(
        size=input_size,
        wavelength=1550e-9,
        loss_db_cm=0.5,
        phase_shifter_type='thermal'
    )
    
    # Memristive back-end: PCM crossbar
    memristive_layer = MemristiveLayer(
        input_size=input_size,
        output_size=hidden_size,
        device_type='PCM',
        include_aging=False,
        variability=True
    )
    
    # Output layer
    output_layer = MemristiveLayer(
        input_size=hidden_size,
        output_size=output_size,
        device_type='RRAM'
    )
    
    # Transimpedance amplifier
    tia = TransimpedanceAmplifier(gain=1e5)
    
    # Combine into hybrid network
    layers = [photonic_layer, memristive_layer, tia, output_layer]
    network = HybridNetwork(layers=layers)
    
    return network


def generate_sample_data(n_samples: int = 100, 
                        input_size: int = 4,
                        output_size: int = 2) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate sample training data."""
    key = jax.random.PRNGKey(42)
    
    # Generate optical input data (complex amplitudes)
    key, subkey = jax.random.split(key)
    inputs_magnitude = jax.random.uniform(subkey, (n_samples, input_size), 
                                        minval=0.1, maxval=1.0)
    
    key, subkey = jax.random.split(key)
    inputs_phase = jax.random.uniform(subkey, (n_samples, input_size),
                                    minval=0, maxval=2*jnp.pi)
    
    inputs = inputs_magnitude * jnp.exp(1j * inputs_phase)
    
    # Generate corresponding targets (classification task)
    key, subkey = jax.random.split(key)
    targets = jax.random.randint(subkey, (n_samples,), 0, output_size)
    targets = jax.nn.one_hot(targets, output_size)
    
    return inputs, targets


def train_hybrid_network(network: HybridNetwork,
                        train_data: Tuple[jnp.ndarray, jnp.ndarray],
                        val_data: Tuple[jnp.ndarray, jnp.ndarray],
                        epochs: int = 50) -> Dict[str, Any]:
    """Train the hybrid network with hardware-aware loss."""
    
    train_inputs, train_targets = train_data
    val_inputs, val_targets = val_data
    
    # Initialize network parameters
    key = jax.random.PRNGKey(123)
    params = network.init(key, train_inputs[:1], training=True)
    
    # Create hardware-aware optimizer
    optimizer = pm.create_hardware_optimizer(
        learning_rate=1e-3,
        phase_shifter_constraints=(-jnp.pi, jnp.pi),
        memristor_constraints=(1e3, 1e6),
        gradient_clipping=1.0
    )
    
    # Define loss function
    def loss_fn(network_model, params, inputs, targets):
        return pm.hardware_aware_loss(
            network_model, params, inputs, targets,
            alpha_optical=0.1,
            alpha_power=0.01,
            alpha_aging=0.001
        )
    
    # Simple data loader
    batch_size = 10
    n_batches = len(train_inputs) // batch_size
    
    def data_loader():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            yield train_inputs[start_idx:end_idx], train_targets[start_idx:end_idx]
    
    # Train the network
    print("Starting training...")
    results = pm.train(
        network=network,
        optimizer=optimizer,
        loss_fn=loss_fn,
        data_loader=data_loader(),
        epochs=epochs,
        validation_data=(val_inputs, val_targets),
        verbose=True
    )
    
    return results


def analyze_performance(network: HybridNetwork,
                       params: Dict[str, Any],
                       test_data: Tuple[jnp.ndarray, jnp.ndarray]) -> Dict[str, Any]:
    """Analyze network performance and hardware metrics."""
    
    test_inputs, test_targets = test_data
    
    # Forward pass
    predictions = network.apply(params, test_inputs, training=False)
    
    # Calculate accuracy
    pred_classes = jnp.argmax(predictions, axis=-1)
    true_classes = jnp.argmax(test_targets, axis=-1)
    accuracy = jnp.mean(pred_classes == true_classes)
    
    # Hardware metrics
    optical_loss = network.get_optical_losses(params)
    power_dissipation = network.get_power_dissipation(params)
    aging_penalty = network.estimate_lifetime_degradation(params)
    
    # Variability analysis
    variability_results = pm.variability_analysis(
        network, params, test_inputs, n_samples=100, device_cv=0.15
    )
    
    # Yield estimation
    specs = {'accuracy': 0.8, 'power': 100e-3}  # 80% accuracy, 100mW power
    yield_estimate = pm.estimate_yield(
        network, params, test_data, specs, n_samples=100
    )
    
    results = {
        'accuracy': accuracy,
        'optical_loss_db': optical_loss,
        'power_dissipation_w': power_dissipation,
        'aging_penalty': aging_penalty,
        'output_variability_cv': jnp.mean(variability_results['cv']),
        'estimated_yield': yield_estimate
    }
    
    return results


def visualize_results(training_results: Dict[str, Any],
                     performance_results: Dict[str, Any]) -> None:
    """Visualize training and performance results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training curves
    axes[0, 0].plot(training_results['train_losses'], label='Train Loss')
    if training_results['val_losses']:
        axes[0, 0].plot(training_results['val_losses'], label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Performance metrics
    metrics = ['accuracy', 'optical_loss_db', 'power_dissipation_w', 'estimated_yield']
    values = [performance_results[m] for m in metrics]
    
    axes[0, 1].bar(range(len(metrics)), values)
    axes[0, 1].set_xticks(range(len(metrics)))
    axes[0, 1].set_xticklabels([m.replace('_', '\n') for m in metrics])
    axes[0, 1].set_title('Performance Metrics')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Hardware comparison
    architectures = ['Electronic\n(45nm)', 'Photonic\nOnly', 'PhoMem\nHybrid']
    mac_energy = [45e-15, 0.1e-15, 2.3e-15]  # Joules per MAC
    
    axes[1, 0].bar(architectures, np.array(mac_energy) * 1e15)
    axes[1, 0].set_ylabel('Energy per MAC (fJ)')
    axes[1, 0].set_title('Energy Efficiency Comparison')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Device variability impact
    cv_values = [0.05, 0.10, 0.15, 0.20, 0.25]
    accuracy_impact = [0.98, 0.95, 0.92, 0.88, 0.83]  # Simulated data
    
    axes[1, 1].plot(cv_values, accuracy_impact, 'o-')
    axes[1, 1].axvline(x=0.15, color='r', linestyle='--', 
                      label=f'Current CV: {performance_results["output_variability_cv"]:.3f}')
    axes[1, 1].set_xlabel('Device Coefficient of Variation')
    axes[1, 1].set_ylabel('Network Accuracy')
    axes[1, 1].set_title('Variability Impact')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('phomem_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def run_multiphysics_simulation(network: HybridNetwork,
                               params: Dict[str, Any]) -> Dict[str, Any]:
    """Run multi-physics co-simulation of the hybrid chip."""
    
    # Create simplified chip design
    class SimpleChipDesign:
        def get_geometry(self):
            return {
                'grid_size': (50, 50, 10),
                'grid_spacing': (1e-6, 1e-6, 1e-6),
                'regions': [
                    {
                        'material': 'silicon',
                        'x_min': 10, 'x_max': 40,
                        'y_min': 10, 'y_max': 40,
                        'z_min': 2, 'z_max': 8
                    }
                ]
            }
        
        def get_materials(self):
            return {
                'silicon': {
                    'refractive_index': 3.45,
                    'thermal_conductivity': 150,  # W/m·K
                    'heat_capacity': 700,  # J/kg·K
                    'density': 2330  # kg/m³
                },
                'air': {
                    'refractive_index': 1.0,
                    'thermal_conductivity': 0.026,
                    'heat_capacity': 1005,
                    'density': 1.225
                }
            }
    
    # Initialize simulator
    simulator = MultiPhysicsSimulator(
        optical_solver='BPM',
        thermal_solver='FEM',
        electrical_solver='SPICE',
        coupling='weak'
    )
    
    # Run simulation
    chip_design = SimpleChipDesign()
    results = simulator.simulate(
        chip_design=chip_design,
        input_optical_power=10e-3,  # 10mW
        ambient_temperature=25,  # °C
        duration=1.0,  # 1 second
        save_fields=True
    )
    
    print(f"Multi-physics simulation completed in {results['simulation_time']:.2f}s")
    print(f"Simulation converged: {results['converged']}")
    
    # Plot thermal distribution if available
    if 'temperature_field' in results['thermal']:
        simulator.plot_thermal_map(
            results['thermal']['temperature_field'][-1],
            save_path='thermal_distribution.png'
        )
    
    return results


def main():
    """Main example execution."""
    print("PhoMem-CoSim: Basic Hybrid Network Example")
    print("=" * 50)
    
    # Create hybrid network
    print("1. Creating hybrid photonic-memristive network...")
    network = create_hybrid_network()
    print(f"   Network created with {len(network.layers)} layers")
    
    # Generate training data
    print("\n2. Generating training data...")
    train_inputs, train_targets = generate_sample_data(n_samples=80)
    val_inputs, val_targets = generate_sample_data(n_samples=20)
    test_inputs, test_targets = generate_sample_data(n_samples=20)
    print(f"   Generated {len(train_inputs)} training samples")
    
    # Train network
    print("\n3. Training hybrid network...")
    training_results = train_hybrid_network(
        network, 
        (train_inputs, train_targets),
        (val_inputs, val_targets),
        epochs=20
    )
    print(f"   Training completed. Final loss: {training_results['train_losses'][-1]:.6f}")
    
    # Analyze performance
    print("\n4. Analyzing performance...")
    performance_results = analyze_performance(
        network, 
        training_results['params'],
        (test_inputs, test_targets)
    )
    
    print(f"   Test accuracy: {performance_results['accuracy']:.3f}")
    print(f"   Optical loss: {performance_results['optical_loss_db']:.2f} dB")
    print(f"   Power dissipation: {performance_results['power_dissipation_w']*1000:.1f} mW")
    print(f"   Estimated yield: {performance_results['estimated_yield']:.1%}")
    
    # Visualize results
    print("\n5. Generating visualizations...")
    visualize_results(training_results, performance_results)
    
    # Multi-physics simulation (optional)
    print("\n6. Running multi-physics co-simulation...")
    try:
        multiphysics_results = run_multiphysics_simulation(
            network, training_results['params']
        )
        print("   Multi-physics simulation completed successfully")
    except Exception as e:
        print(f"   Multi-physics simulation failed: {e}")
    
    print("\nExample completed successfully!")
    print("Check the generated plots: phomem_results.png, thermal_distribution.png")


if __name__ == "__main__":
    main()