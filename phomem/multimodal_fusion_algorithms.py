"""
Advanced Multi-Modal Fusion Algorithms for Photonic-Memristive Systems

Generation 5 Enhancement: Revolutionary fusion of optical, electrical, and quantum
modalities with attention-based architectures and physics-informed constraints.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Union, Callable
import optax
from flax import linen as nn
import numpy as np
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

from .neural.networks import PhotonicLayer, MemristiveLayer
from .photonics.components import MachZehnderInterferometer, PhaseShifter
from .memristors.models import PCMModel, RRAMModel
from .utils.performance import QuantumCoherentProcessor, OpticalFieldAnalyzer


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal fusion algorithms."""
    optical_channels: int = 32
    electrical_channels: int = 64
    quantum_channels: int = 16
    fusion_layers: int = 4
    attention_heads: int = 8
    coherence_preservation: bool = True
    physics_informed_constraints: bool = True
    adaptive_modality_weighting: bool = True
    cross_modal_learning: bool = True


class QuantumCoherentAttention(nn.Module):
    """Quantum-coherent attention mechanism for multi-modal fusion."""
    
    num_heads: int
    quantum_channels: int
    coherence_time: float = 1e-6  # seconds
    
    @nn.compact
    def __call__(
        self, 
        optical_input: jnp.ndarray,
        electrical_input: jnp.ndarray,
        quantum_state: Optional[jnp.ndarray] = None,
        training: bool = False
    ) -> Tuple[jnp.ndarray, Dict]:
        
        batch_size, seq_len = optical_input.shape[:2]
        
        # Convert inputs to quantum-coherent representation
        optical_quantum = self._optical_to_quantum_encoding(optical_input)
        electrical_quantum = self._electrical_to_quantum_encoding(electrical_input)
        
        # Initialize quantum state if not provided
        if quantum_state is None:
            quantum_state = jnp.ones((batch_size, self.quantum_channels), dtype=jnp.complex64) / jnp.sqrt(self.quantum_channels)
        
        # Multi-head quantum attention
        attention_outputs = []
        coherence_metrics = []
        
        for head in range(self.num_heads):
            # Quantum superposition of modalities
            superposition_state = self._create_quantum_superposition(
                optical_quantum, electrical_quantum, quantum_state, head
            )
            
            # Quantum attention computation
            attention_weights, coherence = self._quantum_attention_weights(
                superposition_state, head
            )
            
            # Apply attention with coherence preservation
            attended_output = self._apply_coherent_attention(
                superposition_state, attention_weights, coherence
            )
            
            attention_outputs.append(attended_output)
            coherence_metrics.append(coherence)
        
        # Concatenate multi-head outputs
        multi_head_output = jnp.concatenate(attention_outputs, axis=-1)
        
        # Final projection
        output = nn.Dense(
            features=optical_input.shape[-1],
            name=f"quantum_attention_projection"
        )(multi_head_output)
        
        # Compute attention metrics
        attention_metrics = {
            "average_coherence": jnp.mean(jnp.array(coherence_metrics)),
            "quantum_entanglement": self._measure_entanglement(quantum_state),
            "modal_coupling_strength": self._measure_modal_coupling(optical_quantum, electrical_quantum),
            "decoherence_rate": self._estimate_decoherence_rate(coherence_metrics)
        }
        
        return output, attention_metrics
    
    def _optical_to_quantum_encoding(self, optical_input: jnp.ndarray) -> jnp.ndarray:
        """Encode optical signals as quantum state amplitudes."""
        
        # Normalize optical intensities
        normalized_optical = optical_input / (jnp.max(optical_input) + 1e-8)
        
        # Map to complex amplitudes (phase and magnitude)
        phase = 2 * jnp.pi * normalized_optical[..., :normalized_optical.shape[-1]//2]
        magnitude = normalized_optical[..., normalized_optical.shape[-1]//2:]
        
        # Pad if necessary
        if magnitude.shape[-1] < phase.shape[-1]:
            magnitude = jnp.pad(magnitude, ((0, 0), (0, 0), (0, phase.shape[-1] - magnitude.shape[-1])))
        
        quantum_amplitude = magnitude * jnp.exp(1j * phase)
        
        # Normalize to unit probability
        norm = jnp.sqrt(jnp.sum(jnp.abs(quantum_amplitude)**2, axis=-1, keepdims=True))
        quantum_state = quantum_amplitude / (norm + 1e-8)
        
        return quantum_state
    
    def _electrical_to_quantum_encoding(self, electrical_input: jnp.ndarray) -> jnp.ndarray:
        """Encode electrical signals as quantum state amplitudes."""
        
        # Convert electrical voltages/currents to quantum representation
        # Using voltage-controlled quantum phase encoding
        
        voltage_normalized = jnp.tanh(electrical_input)  # Normalize to [-1, 1]
        
        # Map voltages to quantum phases
        quantum_phase = jnp.pi * voltage_normalized
        
        # Create quantum amplitudes with uniform magnitude and voltage-controlled phase
        magnitude = jnp.ones_like(quantum_phase) / jnp.sqrt(quantum_phase.shape[-1])
        quantum_amplitude = magnitude * jnp.exp(1j * quantum_phase)
        
        return quantum_amplitude
    
    def _create_quantum_superposition(
        self,
        optical_quantum: jnp.ndarray,
        electrical_quantum: jnp.ndarray,
        base_quantum_state: jnp.ndarray,
        head_index: int
    ) -> jnp.ndarray:
        """Create quantum superposition of multi-modal inputs."""
        
        # Head-specific superposition coefficients
        alpha = jnp.cos(jnp.pi * head_index / (2 * self.num_heads))  # Optical weight
        beta = jnp.sin(jnp.pi * head_index / (2 * self.num_heads))   # Electrical weight
        gamma = 0.1  # Base quantum state weight
        
        # Ensure dimensional compatibility
        min_dim = min(optical_quantum.shape[-1], electrical_quantum.shape[-1], base_quantum_state.shape[-1])
        
        optical_trunc = optical_quantum[..., :min_dim]
        electrical_trunc = electrical_quantum[..., :min_dim]
        base_trunc = base_quantum_state[..., :min_dim]
        
        # Create superposition
        superposition = (
            alpha * optical_trunc + 
            beta * electrical_trunc + 
            gamma * base_trunc
        )
        
        # Renormalize
        norm = jnp.sqrt(jnp.sum(jnp.abs(superposition)**2, axis=-1, keepdims=True))
        superposition = superposition / (norm + 1e-8)
        
        return superposition
    
    def _quantum_attention_weights(
        self,
        quantum_state: jnp.ndarray,
        head_index: int
    ) -> Tuple[jnp.ndarray, float]:
        """Compute quantum attention weights and coherence measure."""
        
        # Compute quantum inner products for attention
        # |<ψ_i|ψ_j>|² gives transition probability
        
        batch_size, seq_len, quantum_dim = quantum_state.shape
        
        # Compute pairwise quantum overlaps
        attention_logits = jnp.zeros((batch_size, seq_len, seq_len))
        
        for i in range(seq_len):
            for j in range(seq_len):
                # Quantum overlap |<ψ_i|ψ_j>|²
                overlap = jnp.sum(
                    jnp.conj(quantum_state[:, i, :]) * quantum_state[:, j, :],
                    axis=-1
                )
                attention_logits = attention_logits.at[:, i, j].set(jnp.abs(overlap)**2)
        
        # Apply temperature scaling for head-specific behavior
        temperature = 1.0 + 0.1 * head_index
        attention_weights = nn.softmax(attention_logits / temperature, axis=-1)
        
        # Measure quantum coherence (von Neumann entropy)
        # Higher coherence = lower entropy
        density_matrix = jnp.einsum('bsd,bsd->bss', quantum_state, jnp.conj(quantum_state))
        eigenvalues = jnp.linalg.eigvals(density_matrix)
        eigenvalues = jnp.real(eigenvalues)  # Should be real for density matrix
        eigenvalues = jnp.maximum(eigenvalues, 1e-12)  # Avoid log(0)
        
        von_neumann_entropy = -jnp.sum(eigenvalues * jnp.log(eigenvalues), axis=-1)
        coherence = jnp.exp(-von_neumann_entropy)  # Convert to coherence measure
        avg_coherence = float(jnp.mean(coherence))
        
        return attention_weights, avg_coherence
    
    def _apply_coherent_attention(
        self,
        quantum_state: jnp.ndarray,
        attention_weights: jnp.ndarray,
        coherence: float
    ) -> jnp.ndarray:
        """Apply attention while preserving quantum coherence."""
        
        # Standard attention application
        attended_state = jnp.einsum('bij,bjd->bid', attention_weights, quantum_state)
        
        # Coherence preservation: blend with original state
        coherence_preservation_factor = 0.8 * coherence  # Preserve more when coherence is high
        
        preserved_state = (
            coherence_preservation_factor * quantum_state.mean(axis=1, keepdims=True) +
            (1 - coherence_preservation_factor) * attended_state
        )
        
        # Convert back to real representation for neural network processing
        real_output = jnp.concatenate([
            jnp.real(preserved_state),
            jnp.imag(preserved_state)
        ], axis=-1)
        
        return real_output
    
    def _measure_entanglement(self, quantum_state: jnp.ndarray) -> float:
        """Measure quantum entanglement in the state."""
        
        # Simplified entanglement measure using state purity
        batch_size, quantum_dim = quantum_state.shape
        
        # Compute density matrix
        density_matrix = jnp.einsum('bd,bd->bb', quantum_state, jnp.conj(quantum_state))
        
        # Purity = Tr(ρ²)
        purity = jnp.trace(jnp.matmul(density_matrix, density_matrix))
        
        # Entanglement measure (1 - purity for mixed states)
        entanglement = 1.0 - jnp.real(purity) / batch_size
        
        return float(entanglement)
    
    def _measure_modal_coupling_strength(
        self,
        optical_quantum: jnp.ndarray,
        electrical_quantum: jnp.ndarray
    ) -> float:
        """Measure coupling strength between optical and electrical modalities."""
        
        # Compute cross-correlation in quantum domain
        min_dim = min(optical_quantum.shape[-1], electrical_quantum.shape[-1])
        
        optical_trunc = optical_quantum[..., :min_dim]
        electrical_trunc = electrical_quantum[..., :min_dim]
        
        # Quantum fidelity between modalities
        fidelity = jnp.abs(jnp.sum(
            jnp.conj(optical_trunc) * electrical_trunc,
            axis=-1
        ))**2
        
        coupling_strength = float(jnp.mean(fidelity))
        
        return coupling_strength
    
    def _estimate_decoherence_rate(self, coherence_metrics: List[float]) -> float:
        """Estimate quantum decoherence rate from coherence evolution."""
        
        if len(coherence_metrics) < 2:
            return 0.0
        
        # Simple exponential decay model
        coherence_decay = np.gradient(coherence_metrics)
        avg_decay_rate = float(np.mean(np.abs(coherence_decay)))
        
        # Convert to decoherence time (inverse rate)
        decoherence_rate = avg_decay_rate / self.coherence_time
        
        return decoherence_rate


class PhysicsInformedFusion(nn.Module):
    """Physics-informed fusion layer with conservation laws."""
    
    fusion_dim: int
    enforce_energy_conservation: bool = True
    enforce_momentum_conservation: bool = True
    optical_wavelength: float = 1550e-9  # meters
    
    @nn.compact
    def __call__(
        self,
        optical_features: jnp.ndarray,
        electrical_features: jnp.ndarray,
        quantum_features: jnp.ndarray,
        training: bool = False
    ) -> Tuple[jnp.ndarray, Dict]:
        
        # Project all modalities to fusion dimension
        optical_proj = nn.Dense(
            features=self.fusion_dim,
            name="optical_projection"
        )(optical_features)
        
        electrical_proj = nn.Dense(
            features=self.fusion_dim,
            name="electrical_projection"
        )(electrical_features)
        
        quantum_proj = nn.Dense(
            features=self.fusion_dim,
            name="quantum_projection"
        )(quantum_features)
        
        # Physics-informed fusion with conservation laws
        fused_features, physics_metrics = self._physics_informed_combination(
            optical_proj, electrical_proj, quantum_proj
        )
        
        # Apply physics-constrained transformation
        if self.enforce_energy_conservation:
            fused_features = self._enforce_energy_conservation(fused_features)
        
        if self.enforce_momentum_conservation:
            fused_features = self._enforce_momentum_conservation(fused_features)
        
        # Final fusion layer
        output = nn.Dense(
            features=self.fusion_dim,
            name="physics_fusion_output"
        )(fused_features)
        
        return output, physics_metrics
    
    def _physics_informed_combination(
        self,
        optical: jnp.ndarray,
        electrical: jnp.ndarray,
        quantum: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Dict]:
        """Combine modalities with physics-based rules."""
        
        # Optical-electrical coupling via photoelectric effect
        photoelectric_coupling = self._photoelectric_interaction(optical, electrical)
        
        # Quantum-classical coupling
        quantum_classical_coupling = self._quantum_classical_interaction(quantum, electrical)
        
        # Optical-quantum coupling via quantum optics
        optical_quantum_coupling = self._optical_quantum_interaction(optical, quantum)
        
        # Weighted combination with physics-based coefficients
        alpha_oe = 0.4  # Optical-electrical coupling strength
        alpha_qc = 0.3  # Quantum-classical coupling strength
        alpha_oq = 0.3  # Optical-quantum coupling strength
        
        fused = (
            alpha_oe * photoelectric_coupling +
            alpha_qc * quantum_classical_coupling +
            alpha_oq * optical_quantum_coupling
        )
        
        # Compute physics metrics
        physics_metrics = {
            "photoelectric_efficiency": self._compute_photoelectric_efficiency(optical, electrical),
            "quantum_coupling_strength": self._compute_quantum_coupling(quantum, electrical),
            "optical_quantum_correlation": self._compute_optical_quantum_correlation(optical, quantum),
            "total_energy": self._compute_total_energy(fused),
            "energy_conservation_violation": self._check_energy_conservation(optical, electrical, quantum, fused)
        }
        
        return fused, physics_metrics
    
    def _photoelectric_interaction(
        self,
        optical: jnp.ndarray,
        electrical: jnp.ndarray
    ) -> jnp.ndarray:
        """Model photoelectric effect coupling."""
        
        # Photon energy: E = hc/λ
        h = 6.626e-34  # Planck constant
        c = 3e8        # Speed of light
        photon_energy = h * c / self.optical_wavelength
        
        # Photoelectric current proportional to optical intensity
        # I = η * P * e / (h*f), where η is quantum efficiency
        quantum_efficiency = 0.8
        
        # Normalize optical power and convert to photoelectric response
        optical_normalized = nn.tanh(optical)  # Saturate at high intensities
        photoelectric_current = quantum_efficiency * optical_normalized
        
        # Combine with existing electrical signal
        combined = electrical + photoelectric_current
        
        return combined
    
    def _quantum_classical_interaction(
        self,
        quantum: jnp.ndarray,
        electrical: jnp.ndarray
    ) -> jnp.ndarray:
        """Model quantum-classical interface."""
        
        # Quantum measurement induces classical electrical signal
        # |ψ|² → classical probability → electrical signal
        
        quantum_probability = jnp.abs(quantum)**2
        
        # Convert quantum probability to electrical measurement signal
        measurement_signal = jnp.sqrt(quantum_probability)  # Shot noise scaling
        
        # Gate electrical signal with quantum measurement
        gated_electrical = electrical * (1 + 0.1 * measurement_signal)
        
        return gated_electrical
    
    def _optical_quantum_interaction(
        self,
        optical: jnp.ndarray,
        quantum: jnp.ndarray
    ) -> jnp.ndarray:
        """Model optical-quantum coupling via cavity QED."""
        
        # Strong coupling regime: g >> (κ, γ)
        # where g is coupling strength, κ is cavity decay, γ is atomic decay
        
        coupling_strength = 0.5
        
        # Rabi oscillations between optical field and quantum state
        rabi_frequency = coupling_strength * optical
        
        # Quantum state evolution under optical drive
        driven_quantum = quantum * jnp.cos(rabi_frequency) + optical * jnp.sin(rabi_frequency)
        
        return driven_quantum
    
    def _enforce_energy_conservation(self, features: jnp.ndarray) -> jnp.ndarray:
        """Enforce energy conservation in feature space."""
        
        # Compute total energy (sum of squares)
        total_energy = jnp.sum(features**2, axis=-1, keepdims=True)
        
        # Normalize to conserve energy
        energy_normalized = features / jnp.sqrt(total_energy + 1e-8)
        
        return energy_normalized
    
    def _enforce_momentum_conservation(self, features: jnp.ndarray) -> jnp.ndarray:
        """Enforce momentum conservation (simplified as sum conservation)."""
        
        # Ensure total momentum (sum) is conserved
        total_momentum = jnp.sum(features, axis=-1, keepdims=True)
        
        # Apply momentum conservation constraint
        momentum_conserved = features - total_momentum / features.shape[-1]
        
        return momentum_conserved
    
    def _compute_photoelectric_efficiency(
        self,
        optical: jnp.ndarray,
        electrical: jnp.ndarray
    ) -> float:
        """Compute photoelectric conversion efficiency."""
        
        optical_power = jnp.mean(jnp.sum(optical**2, axis=-1))
        electrical_power = jnp.mean(jnp.sum(electrical**2, axis=-1))
        
        efficiency = electrical_power / (optical_power + 1e-8)
        
        return float(efficiency)
    
    def _compute_quantum_coupling(
        self,
        quantum: jnp.ndarray,
        electrical: jnp.ndarray
    ) -> float:
        """Compute quantum-classical coupling strength."""
        
        # Cross-correlation between quantum and electrical signals
        quantum_flat = quantum.reshape(-1)
        electrical_flat = electrical.reshape(-1)
        
        correlation = jnp.corrcoef(
            jnp.stack([jnp.real(quantum_flat), electrical_flat])
        )[0, 1]
        
        return float(jnp.abs(correlation))
    
    def _compute_optical_quantum_correlation(
        self,
        optical: jnp.ndarray,
        quantum: jnp.ndarray
    ) -> float:
        """Compute optical-quantum correlation."""
        
        optical_flat = optical.reshape(-1)
        quantum_flat = jnp.real(quantum).reshape(-1)
        
        correlation = jnp.corrcoef(
            jnp.stack([optical_flat, quantum_flat])
        )[0, 1]
        
        return float(jnp.abs(correlation))
    
    def _compute_total_energy(self, features: jnp.ndarray) -> float:
        """Compute total energy in feature space."""
        
        total_energy = jnp.sum(features**2)
        
        return float(total_energy)
    
    def _check_energy_conservation(
        self,
        optical: jnp.ndarray,
        electrical: jnp.ndarray,
        quantum: jnp.ndarray,
        fused: jnp.ndarray
    ) -> float:
        """Check energy conservation violation."""
        
        input_energy = (
            jnp.sum(optical**2) +
            jnp.sum(electrical**2) +
            jnp.sum(jnp.abs(quantum)**2)
        )
        
        output_energy = jnp.sum(fused**2)
        
        violation = jnp.abs(output_energy - input_energy) / (input_energy + 1e-8)
        
        return float(violation)


class AdaptiveModalityWeighting(nn.Module):
    """Adaptive weighting of different modalities based on signal quality."""
    
    num_modalities: int
    adaptation_rate: float = 0.1
    
    @nn.compact
    def __call__(
        self,
        modal_features: List[jnp.ndarray],
        modal_qualities: Optional[List[float]] = None,
        training: bool = False
    ) -> Tuple[jnp.ndarray, Dict]:
        
        if modal_qualities is None:
            # Estimate modal qualities from signal characteristics
            modal_qualities = [self._estimate_signal_quality(features) for features in modal_features]
        
        # Compute adaptive weights
        quality_weights = nn.softmax(jnp.array(modal_qualities), axis=0)
        
        # Learn complementary weights
        learned_weights = self.param(
            'modality_weights',
            nn.initializers.ones,
            (self.num_modalities,)
        )
        learned_weights = nn.softmax(learned_weights, axis=0)
        
        # Combine quality-based and learned weights
        final_weights = (
            (1 - self.adaptation_rate) * learned_weights +
            self.adaptation_rate * quality_weights
        )
        
        # Apply weighted combination
        weighted_features = []
        for i, features in enumerate(modal_features):
            weighted = final_weights[i] * features
            weighted_features.append(weighted)
        
        # Concatenate weighted modalities
        combined_features = jnp.concatenate(weighted_features, axis=-1)
        
        # Final fusion layer
        output = nn.Dense(
            features=modal_features[0].shape[-1],
            name="adaptive_fusion_output"
        )(combined_features)
        
        weighting_metrics = {
            "quality_weights": quality_weights,
            "learned_weights": learned_weights,
            "final_weights": final_weights,
            "weight_entropy": -jnp.sum(final_weights * jnp.log(final_weights + 1e-8)),
            "dominant_modality": int(jnp.argmax(final_weights))
        }
        
        return output, weighting_metrics
    
    def _estimate_signal_quality(self, features: jnp.ndarray) -> float:
        """Estimate signal quality from statistical properties."""
        
        # Signal-to-noise ratio estimate
        signal_power = jnp.var(features)
        noise_estimate = jnp.mean(jnp.abs(jnp.diff(features, axis=-1)))
        
        snr = signal_power / (noise_estimate + 1e-8)
        
        # Normalize to [0, 1] range
        quality = jnp.tanh(snr / 10.0)
        
        return float(quality)


class MultiModalFusionNetwork(nn.Module):
    """Complete multi-modal fusion network architecture."""
    
    config: MultiModalConfig
    
    @nn.compact
    def __call__(
        self,
        optical_input: jnp.ndarray,
        electrical_input: jnp.ndarray,
        quantum_input: Optional[jnp.ndarray] = None,
        training: bool = False
    ) -> Tuple[jnp.ndarray, Dict]:
        
        batch_size = optical_input.shape[0]
        
        # Initialize quantum input if not provided
        if quantum_input is None:
            quantum_input = jnp.ones(
                (batch_size, self.config.quantum_channels),
                dtype=jnp.complex64
            ) / jnp.sqrt(self.config.quantum_channels)
        
        # Multi-layer fusion processing
        current_optical = optical_input
        current_electrical = electrical_input
        current_quantum = quantum_input
        
        all_metrics = {}
        
        for layer_idx in range(self.config.fusion_layers):
            
            # Quantum-coherent attention
            if self.config.cross_modal_learning:
                attended_output, attention_metrics = QuantumCoherentAttention(
                    num_heads=self.config.attention_heads,
                    quantum_channels=self.config.quantum_channels,
                    name=f"quantum_attention_{layer_idx}"
                )(
                    current_optical,
                    current_electrical,
                    current_quantum,
                    training=training
                )
                
                all_metrics[f"attention_layer_{layer_idx}"] = attention_metrics
                current_optical = attended_output
            
            # Physics-informed fusion
            if self.config.physics_informed_constraints:
                fused_output, physics_metrics = PhysicsInformedFusion(
                    fusion_dim=current_optical.shape[-1],
                    name=f"physics_fusion_{layer_idx}"
                )(
                    current_optical,
                    current_electrical,
                    current_quantum,
                    training=training
                )
                
                all_metrics[f"physics_layer_{layer_idx}"] = physics_metrics
                current_optical = fused_output
            
            # Adaptive modality weighting
            if self.config.adaptive_modality_weighting:
                modal_features = [current_optical, current_electrical, jnp.real(current_quantum)]
                
                weighted_output, weighting_metrics = AdaptiveModalityWeighting(
                    num_modalities=len(modal_features),
                    name=f"adaptive_weighting_{layer_idx}"
                )(modal_features, training=training)
                
                all_metrics[f"weighting_layer_{layer_idx}"] = weighting_metrics
                current_optical = weighted_output
            
            # Update states for next layer
            if layer_idx < self.config.fusion_layers - 1:
                # Project to next layer dimensions
                current_electrical = nn.Dense(
                    features=current_optical.shape[-1],
                    name=f"electrical_projection_{layer_idx}"
                )(current_electrical)
                
                # Update quantum state (simplified evolution)
                current_quantum = current_quantum * jnp.exp(1j * 0.1 * layer_idx)
        
        # Final output projection
        final_output = nn.Dense(
            features=optical_input.shape[-1],
            name="final_multimodal_output"
        )(current_optical)
        
        # Aggregate metrics across layers
        aggregated_metrics = self._aggregate_layer_metrics(all_metrics)
        
        return final_output, aggregated_metrics
    
    def _aggregate_layer_metrics(self, all_metrics: Dict) -> Dict:
        """Aggregate metrics across all fusion layers."""
        
        aggregated = {
            "avg_coherence": 0.0,
            "avg_entanglement": 0.0,
            "avg_energy_conservation_violation": 0.0,
            "total_decoherence_rate": 0.0,
            "modality_dominance": {},
            "physics_compliance_score": 0.0
        }
        
        num_attention_layers = 0
        num_physics_layers = 0
        
        for layer_name, metrics in all_metrics.items():
            if "attention" in layer_name:
                num_attention_layers += 1
                aggregated["avg_coherence"] += metrics.get("average_coherence", 0.0)
                aggregated["avg_entanglement"] += metrics.get("quantum_entanglement", 0.0)
                aggregated["total_decoherence_rate"] += metrics.get("decoherence_rate", 0.0)
            
            elif "physics" in layer_name:
                num_physics_layers += 1
                aggregated["avg_energy_conservation_violation"] += metrics.get("energy_conservation_violation", 0.0)
            
            elif "weighting" in layer_name:
                dominant_modality = metrics.get("dominant_modality", -1)
                if dominant_modality >= 0:
                    modality_name = ["optical", "electrical", "quantum"][dominant_modality]
                    aggregated["modality_dominance"][modality_name] = aggregated["modality_dominance"].get(modality_name, 0) + 1
        
        # Compute averages
        if num_attention_layers > 0:
            aggregated["avg_coherence"] /= num_attention_layers
            aggregated["avg_entanglement"] /= num_attention_layers
        
        if num_physics_layers > 0:
            aggregated["avg_energy_conservation_violation"] /= num_physics_layers
        
        # Physics compliance score (higher is better)
        aggregated["physics_compliance_score"] = 1.0 - min(1.0, aggregated["avg_energy_conservation_violation"])
        
        return aggregated


# Example usage and benchmarking
if __name__ == "__main__":
    
    # Configuration
    config = MultiModalConfig(
        optical_channels=64,
        electrical_channels=32,
        quantum_channels=16,
        fusion_layers=3,
        attention_heads=4,
        coherence_preservation=True,
        physics_informed_constraints=True,
        adaptive_modality_weighting=True,
        cross_modal_learning=True
    )
    
    # Create model
    model = MultiModalFusionNetwork(config=config)
    
    # Initialize parameters
    rng_key = jax.random.PRNGKey(42)
    batch_size = 8
    seq_len = 16
    
    # Generate dummy multi-modal inputs
    optical_input = jax.random.normal(rng_key, (batch_size, seq_len, config.optical_channels))
    electrical_input = jax.random.normal(rng_key, (batch_size, seq_len, config.electrical_channels))
    
    # Initialize model parameters
    params = model.init(
        rng_key,
        optical_input,
        electrical_input,
        training=True
    )
    
    # Forward pass
    output, metrics = model.apply(
        params,
        optical_input,
        electrical_input,
        training=True
    )
    
    print(f"Multi-modal fusion completed!")
    print(f"Output shape: {output.shape}")
    print(f"Average coherence: {metrics['avg_coherence']:.4f}")
    print(f"Average entanglement: {metrics['avg_entanglement']:.4f}")
    print(f"Physics compliance score: {metrics['physics_compliance_score']:.4f}")
    print(f"Modality dominance: {metrics['modality_dominance']}")
    
    # Benchmark performance
    import time
    
    num_trials = 100
    start_time = time.time()
    
    for _ in range(num_trials):
        output, _ = model.apply(
            params,
            optical_input,
            electrical_input,
            training=False
        )
    
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / num_trials
    
    print(f"Average inference time: {avg_inference_time*1000:.2f} ms")
    print(f"Throughput: {batch_size/avg_inference_time:.1f} samples/sec")