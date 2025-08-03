"""
Training utilities for hybrid photonic-memristive networks.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Callable, Tuple, Optional
import chex
from functools import partial
import optax
from tqdm import tqdm

from .networks import HybridNetwork


def hardware_aware_loss(network: HybridNetwork,
                       params: Dict[str, Any],
                       inputs: chex.Array,
                       targets: chex.Array,
                       alpha_optical: float = 0.1,
                       alpha_power: float = 0.01,
                       alpha_aging: float = 0.001) -> float:
    """
    Hardware-aware loss function including device constraints.
    
    Args:
        network: Hybrid network model
        params: Network parameters
        inputs: Training inputs
        targets: Training targets
        alpha_optical: Weight for optical loss penalty
        alpha_power: Weight for power dissipation penalty
        alpha_aging: Weight for aging penalty
        
    Returns:
        Total loss including hardware penalties
    """
    # Forward pass to get predictions
    predictions = network.apply(params, inputs, training=True)
    
    # Task loss (MSE)
    task_loss = jnp.mean((predictions - targets)**2)
    
    # Hardware penalties
    optical_loss = network.get_optical_losses(params)
    power_dissipation = network.get_power_dissipation(params)
    aging_penalty = network.estimate_lifetime_degradation(params)
    
    # Combined loss
    total_loss = (task_loss + 
                  alpha_optical * optical_loss +
                  alpha_power * power_dissipation +
                  alpha_aging * aging_penalty)
    
    return total_loss


def create_hardware_optimizer(learning_rate: float = 1e-3,
                            phase_shifter_constraints: Tuple[float, float] = (-jnp.pi, jnp.pi),
                            memristor_constraints: Tuple[float, float] = (1e3, 1e6),
                            gradient_clipping: float = 1.0) -> optax.GradientTransformation:
    """
    Create optimizer with hardware-specific constraints.
    
    Args:
        learning_rate: Base learning rate
        phase_shifter_constraints: (min, max) phase values
        memristor_constraints: (min, max) resistance values in Ohms
        gradient_clipping: Gradient clipping threshold
        
    Returns:
        Optax optimizer with constraints
    """
    # Base optimizer
    optimizer = optax.adam(learning_rate)
    
    # Add gradient clipping
    if gradient_clipping > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(gradient_clipping),
            optimizer
        )
    
    # Add parameter constraints
    def constrain_params(updates, params):
        """Apply hardware constraints to parameter updates."""
        constrained_updates = {}
        
        for module, module_params in updates.items():
            constrained_module = {}
            
            for param_name, param_updates in module_params.items():
                if 'phases' in param_name:
                    # Constrain phase shifters
                    new_phases = params[module][param_name] + param_updates
                    constrained_phases = jnp.clip(
                        new_phases, 
                        phase_shifter_constraints[0],
                        phase_shifter_constraints[1]
                    )
                    constrained_module[param_name] = constrained_phases - params[module][param_name]
                
                elif 'states' in param_name:
                    # Constrain memristor states (0 to 1)
                    new_states = params[module][param_name] + param_updates
                    constrained_states = jnp.clip(new_states, 0.0, 1.0)
                    constrained_module[param_name] = constrained_states - params[module][param_name]
                
                else:
                    # No constraints for other parameters
                    constrained_module[param_name] = param_updates
            
            constrained_updates[module] = constrained_module
        
        return constrained_updates
    
    # Add constraint transform
    constraint_transform = optax.stateless(constrain_params)
    optimizer = optax.chain(optimizer, constraint_transform)
    
    return optimizer


def train(network: HybridNetwork,
          optimizer: optax.GradientTransformation,
          loss_fn: Callable,
          data_loader: Any,
          epochs: int = 100,
          validation_data: Optional[Tuple[chex.Array, chex.Array]] = None,
          checkpoint_every: int = 10,
          verbose: bool = True) -> Dict[str, Any]:
    """
    Train hybrid network with hardware-aware optimization.
    
    Args:
        network: Network to train
        optimizer: Optax optimizer
        loss_fn: Loss function
        data_loader: Training data iterator
        epochs: Number of training epochs
        validation_data: Optional validation (inputs, targets)
        checkpoint_every: Checkpoint frequency
        verbose: Whether to print progress
        
    Returns:
        Training results dictionary
    """
    # Initialize parameters
    key = jax.random.PRNGKey(42)
    sample_input = next(iter(data_loader))[0][:1]  # Single batch element
    params = network.init(key, sample_input, training=True)
    
    # Initialize optimizer state
    opt_state = optimizer.init(params)
    
    # Training metrics
    train_losses = []
    val_losses = []
    
    # JIT compile training step
    @jax.jit
    def train_step(params, opt_state, batch):
        inputs, targets = batch
        
        def loss_wrapper(p):
            return loss_fn(network, p, inputs, targets)
        
        loss, grads = jax.value_and_grad(loss_wrapper)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss
    
    # JIT compile validation step
    @jax.jit
    def val_step(params, val_inputs, val_targets):
        return loss_fn(network, params, val_inputs, val_targets)
    
    # Training loop
    if verbose:
        print("Starting training...")
        
    for epoch in range(epochs):
        epoch_losses = []
        
        # Training
        for batch in (tqdm(data_loader, desc=f"Epoch {epoch+1}") if verbose else data_loader):
            params, opt_state, loss = train_step(params, opt_state, batch)
            epoch_losses.append(loss)
        
        train_loss = jnp.mean(jnp.array(epoch_losses))
        train_losses.append(train_loss)
        
        # Validation
        if validation_data is not None:
            val_inputs, val_targets = validation_data
            val_loss = val_step(params, val_inputs, val_targets)
            val_losses.append(val_loss)
        
        # Logging
        if verbose and (epoch + 1) % 5 == 0:
            if validation_data is not None:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}")
    
    results = {
        'params': params,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs': epochs
    }
    
    return results


def train_stdp(snn: Any,
               spike_data: chex.Array,
               learning_rate: float = 1e-2,
               tau_pre: float = 20e-3,
               tau_post: float = 40e-3,
               n_epochs: int = 100) -> Any:
    """
    Train spiking neural network with STDP learning.
    
    Args:
        snn: Photonic SNN model
        spike_data: Spike train data [n_samples, n_steps, input_size]
        learning_rate: STDP learning rate
        tau_pre: Pre-synaptic trace time constant
        tau_post: Post-synaptic trace time constant
        n_epochs: Number of training epochs
        
    Returns:
        Trained SNN with updated synaptic weights
    """
    # Initialize SNN state
    key = jax.random.PRNGKey(42)
    sample_spikes = spike_data[0]  # First sample
    params = snn.init(key, sample_spikes, training=True)
    
    # STDP learning loop
    for epoch in range(n_epochs):
        for sample_idx in range(spike_data.shape[0]):
            spike_train = spike_data[sample_idx]
            
            # Run SNN forward pass with STDP learning enabled
            output_spikes = snn.apply(
                params, 
                spike_train,
                dt=1e-6,
                n_steps=spike_train.shape[0],
                training=True,
                mutable=['stdp_state']
            )
        
        if epoch % 20 == 0:
            print(f"STDP Epoch {epoch}: Learning in progress...")
    
    return snn


def variability_analysis(network: HybridNetwork,
                        params: Dict[str, Any],
                        test_inputs: chex.Array,
                        n_samples: int = 1000,
                        device_cv: float = 0.15) -> Dict[str, chex.Array]:
    """
    Monte Carlo analysis of device variability effects.
    
    Args:
        network: Network to analyze
        params: Nominal network parameters
        test_inputs: Test input data
        n_samples: Number of Monte Carlo samples
        device_cv: Device coefficient of variation
        
    Returns:
        Variability analysis results
    """
    key = jax.random.PRNGKey(123)
    outputs = []
    
    for sample in range(n_samples):
        key, subkey = jax.random.split(key)
        
        # Add device variations to parameters
        varied_params = add_device_variations(params, subkey, device_cv)
        
        # Forward pass with varied parameters
        output = network.apply(varied_params, test_inputs, training=False)
        outputs.append(output)
    
    outputs = jnp.stack(outputs)
    
    # Calculate statistics
    mean_output = jnp.mean(outputs, axis=0)
    std_output = jnp.std(outputs, axis=0)
    
    results = {
        'mean': mean_output,
        'std': std_output,
        'cv': std_output / (mean_output + 1e-8),
        'all_outputs': outputs
    }
    
    return results


def add_device_variations(params: Dict[str, Any],
                         key: chex.PRNGKey,
                         cv: float = 0.15) -> Dict[str, Any]:
    """Add realistic device variations to network parameters."""
    varied_params = {}
    
    for module, module_params in params.items():
        varied_module = {}
        
        for param_name, param_values in module_params.items():
            key, subkey = jax.random.split(key)
            
            if 'states' in param_name:
                # Memristor conductance variations (log-normal)
                sigma = jnp.sqrt(jnp.log(1 + cv**2))
                variations = jax.random.lognormal(subkey, sigma, param_values.shape)
                variations = variations / jnp.mean(variations)  # Normalize
                varied_module[param_name] = param_values * variations
            
            elif 'phases' in param_name:
                # Phase shifter variations (normal)
                phase_std = cv * jnp.pi  # Phase variation
                variations = jax.random.normal(subkey, param_values.shape) * phase_std
                varied_module[param_name] = param_values + variations
            
            else:
                # No variations for other parameters
                varied_module[param_name] = param_values
        
        varied_params[module] = varied_module
    
    return varied_params


def estimate_yield(network: HybridNetwork,
                  params: Dict[str, Any],
                  test_data: Tuple[chex.Array, chex.Array],
                  specs: Dict[str, float],
                  n_samples: int = 1000) -> float:
    """
    Estimate manufacturing yield based on performance specifications.
    
    Args:
        network: Network model
        params: Nominal parameters
        test_data: Test (inputs, targets)
        specs: Performance specifications {'accuracy': min_acc, 'power': max_power}
        n_samples: Number of Monte Carlo samples
        
    Returns:
        Estimated yield fraction
    """
    test_inputs, test_targets = test_data
    
    # Run variability analysis
    variability_results = variability_analysis(
        network, params, test_inputs, n_samples
    )
    
    passing_samples = 0
    
    for sample_idx in range(n_samples):
        sample_output = variability_results['all_outputs'][sample_idx]
        
        # Calculate sample metrics
        accuracy = calculate_accuracy(sample_output, test_targets)
        power = network.get_power_dissipation(params)  # Simplified
        
        # Check specifications
        meets_accuracy = accuracy >= specs.get('accuracy', 0.0)
        meets_power = power <= specs.get('power', float('inf'))
        
        if meets_accuracy and meets_power:
            passing_samples += 1
    
    yield_estimate = passing_samples / n_samples
    return yield_estimate


def calculate_accuracy(predictions: chex.Array, 
                      targets: chex.Array) -> float:
    """Calculate classification accuracy."""
    if predictions.ndim == 2 and targets.ndim == 2:
        # Multi-class classification
        pred_classes = jnp.argmax(predictions, axis=-1)
        true_classes = jnp.argmax(targets, axis=-1)
        return jnp.mean(pred_classes == true_classes)
    else:
        # Binary classification or regression
        return 1.0 - jnp.mean((predictions - targets)**2)  # 1 - MSE as proxy