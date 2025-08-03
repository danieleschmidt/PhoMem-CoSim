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
    """Plot network performance metrics."""
    pass


def plot_device_characteristics(device, save_path: Optional[str] = None):
    """Plot device I-V characteristics."""
    pass