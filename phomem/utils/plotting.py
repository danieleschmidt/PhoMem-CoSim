"""
Plotting utilities for visualization.
"""

import numpy as np
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def plot_thermal_map(temperature_field, save_path: Optional[str] = None):
    """Plot thermal distribution."""
    if temperature_field.ndim == 3:
        temp_2d = temperature_field[:, :, temperature_field.shape[2]//2]
    else:
        temp_2d = temperature_field
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(temp_2d.T, cmap='hot', origin='lower')
    plt.colorbar(im, label='Temperature (K)')
    plt.title('Temperature Distribution')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_optical_field(field, save_path: Optional[str] = None):
    """Plot optical field distribution."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(np.abs(field), cmap='viridis')
    plt.title('Field Amplitude')
    plt.colorbar()
    
    plt.subplot(132)
    plt.imshow(np.angle(field), cmap='hsv')
    plt.title('Field Phase')
    plt.colorbar()
    
    plt.subplot(133)
    plt.imshow(np.abs(field)**2, cmap='hot')
    plt.title('Intensity')
    plt.colorbar()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_network_performance(results: Dict[str, Any], save_path: Optional[str] = None):
    """Plot comprehensive network performance metrics."""
    try:
        import seaborn as sns
        colors = sns.color_palette("husl", 10)
    except ImportError:
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training curves
    if 'train_losses' in results:
        axes[0, 0].plot(results['train_losses'], label='Train Loss', 
                      color=colors[0], linewidth=2)
        if 'val_losses' in results and results['val_losses']:
            axes[0, 0].plot(results['val_losses'], label='Validation Loss', 
                          color=colors[1], linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy vs Hardware Metrics
    if 'pareto_front' in results and results['pareto_front']:
        pareto_data = results['pareto_front']
        accuracies = [sol['metrics']['accuracy'] for sol in pareto_data]
        powers = [sol['metrics']['power_mw'] for sol in pareto_data]
        
        scatter = axes[0, 1].scatter(powers, accuracies, 
                                   c=range(len(accuracies)), 
                                   cmap='viridis', s=100, alpha=0.7)
        axes[0, 1].set_xlabel('Power (mW)')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy vs Power Trade-off (Pareto Front)')
        plt.colorbar(scatter, ax=axes[0, 1], label='Solution Index')
    
    # Device variability impact
    if 'variability_analysis' in results:
        var_data = results['variability_analysis']
        cv_values = var_data.get('cv_range', [0.05, 0.1, 0.15, 0.2, 0.25])
        accuracies = var_data.get('accuracy_impact', [0.98, 0.95, 0.92, 0.88, 0.83])
        
        axes[0, 2].plot(cv_values, accuracies, 'o-', color=colors[2], 
                      linewidth=2, markersize=8)
        axes[0, 2].set_xlabel('Coefficient of Variation')
        axes[0, 2].set_ylabel('Network Accuracy')
        axes[0, 2].set_title('Variability Impact Analysis')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Power breakdown
    if 'power_breakdown' in results:
        power_data = results['power_breakdown']
        components = list(power_data.keys())
        powers = list(power_data.values())
        
        axes[1, 0].pie(powers, labels=components, autopct='%1.1f%%', 
                      colors=colors[:len(components)])
        axes[1, 0].set_title('Power Consumption Breakdown')
    
    # Optical loss distribution
    if 'optical_analysis' in results:
        loss_data = results['optical_analysis']
        wavelengths = loss_data.get('wavelength', np.linspace(1520, 1580, 100))
        transmission = loss_data.get('transmission', np.ones_like(wavelengths))
        
        axes[1, 1].plot(wavelengths, 10*np.log10(transmission + 1e-10), 
                      color=colors[3], linewidth=2)
        axes[1, 1].set_xlabel('Wavelength (nm)')
        axes[1, 1].set_ylabel('Transmission (dB)')
        axes[1, 1].set_title('Optical Transmission Spectrum')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Performance comparison
    if 'benchmarks' in results:
        benchmark_data = results['benchmarks']
        architectures = list(benchmark_data.keys())
        metrics = ['Energy', 'Latency', 'Area']
        
        x = np.arange(len(architectures))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [benchmark_data[arch].get(metric.lower(), 0) 
                     for arch in architectures]
            axes[1, 2].bar(x + i*width, values, width, 
                          label=metric, color=colors[i])
        
        axes[1, 2].set_xlabel('Architecture')
        axes[1, 2].set_ylabel('Normalized Performance')
        axes[1, 2].set_title('Architecture Comparison')
        axes[1, 2].set_xticks(x + width)
        axes[1, 2].set_xticklabels(architectures, rotation=45)
        axes[1, 2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_device_characteristics(device_data: Dict[str, Any], save_path: Optional[str] = None):
    """Plot detailed device I-V and switching characteristics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    if 'iv_curves' in device_data:
        iv_data = device_data['iv_curves']
        voltages = iv_data['voltage']
        currents = iv_data['current']
        
        # Linear I-V plot
        axes[0, 0].plot(voltages, currents, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Voltage (V)')
        axes[0, 0].set_ylabel('Current (A)')
        axes[0, 0].set_title('I-V Characteristics')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Log-scale I-V plot
        positive_current = np.abs(currents) + 1e-12
        axes[0, 1].semilogy(voltages, positive_current, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Voltage (V)')
        axes[0, 1].set_ylabel('|Current| (A)')
        axes[0, 1].set_title('I-V Characteristics (Log Scale)')
        axes[0, 1].grid(True, alpha=0.3)
    
    if 'switching_dynamics' in device_data:
        switch_data = device_data['switching_dynamics']
        time = switch_data['time']
        voltage = switch_data['voltage']
        current = switch_data['current']
        
        # Switching dynamics plot
        ax_twin = axes[1, 0].twinx()
        line1 = axes[1, 0].plot(time*1e9, voltage, 'g-', label='Voltage', linewidth=2)
        line2 = ax_twin.plot(time*1e9, current*1e6, 'b-', label='Current', linewidth=2)
        
        axes[1, 0].set_xlabel('Time (ns)')
        axes[1, 0].set_ylabel('Voltage (V)', color='g')
        ax_twin.set_ylabel('Current (µA)', color='b')
        axes[1, 0].set_title('Switching Dynamics')
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[1, 0].legend(lines, labels, loc='upper right')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Conductance vs state plot
    if 'state_conductance' in device_data:
        state_data = device_data['state_conductance']
        states = state_data['states']
        conductances = state_data['conductances']
        
        axes[1, 1].plot(states, conductances*1e6, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Device State')
        axes[1, 1].set_ylabel('Conductance (µS)')
        axes[1, 1].set_title('State-Conductance Relationship')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Generate example state-conductance relationship
        states = np.linspace(0, 1, 100)
        # Sigmoid-like relationship
        conductances = 1e-6 + 1e-3 * (1 / (1 + np.exp(-10*(states-0.5))))
        
        axes[1, 1].plot(states, conductances*1e6, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Device State')
        axes[1, 1].set_ylabel('Conductance (µS)')
        axes[1, 1].set_title('State-Conductance Relationship (Model)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_performance_dashboard(results: Dict[str, Any]) -> str:
    """Create comprehensive performance dashboard."""
    plot_network_performance(results)
    
    # Try to create 3D plot if available
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        if 'pareto_front' in results and results['pareto_front']:
            solutions = results['pareto_front']
            accuracies = [sol['metrics']['accuracy'] for sol in solutions]
            powers = [sol['metrics']['power_mw'] for sol in solutions]
            areas = [sol['metrics'].get('area_mm2', 1.0) for sol in solutions]
            
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(powers, areas, accuracies, 
                               c=accuracies, cmap='viridis', s=100)
            
            ax.set_xlabel('Power (mW)')
            ax.set_ylabel('Area (mm²)')
            ax.set_zlabel('Accuracy')
            ax.set_title('3D Pareto Front Visualization')
            
            plt.colorbar(scatter, ax=ax, label='Accuracy')
            plt.show()
            
    except ImportError:
        print("3D plotting not available - install matplotlib with 3D support")
    
    return "Performance dashboard created successfully"