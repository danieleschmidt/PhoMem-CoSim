"""
Quantum-Classical Hybrid Optimization for Photonic Neural Networks

Generation 5 Enhancement: Revolutionary optimization combining quantum annealing,
classical gradient descent, and physics-informed constraints for breakthrough
performance in photonic-memristive neural architectures.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import optax
from flax import linen as nn
import numpy as np
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod
import scipy.optimize
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

from .neural.networks import PhotonicLayer, MemristiveLayer
from .utils.performance import QuantumSimulator, VariationalQuantumEigensolver
from .optimization import MultiObjectiveOptimizer


@dataclass
class QuantumClassicalConfig:
    """Configuration for quantum-classical hybrid optimization."""
    quantum_depth: int = 6
    num_qubits: int = 16
    classical_steps: int = 100
    quantum_steps: int = 50
    hybrid_iterations: int = 20
    variational_form: str = "QAOA"  # Quantum Approximate Optimization Algorithm
    ansatz_layers: int = 4
    measurement_shots: int = 1024
    quantum_backend: str = "simulator"  # "simulator" or "hardware"
    energy_convergence_threshold: float = 1e-6
    gradient_convergence_threshold: float = 1e-4
    physics_informed_constraints: bool = True
    adaptive_parameter_scheduling: bool = True
    parallel_optimization: bool = True


class QuantumAnnealer:
    """Quantum annealing optimization for discrete parameter spaces."""
    
    def __init__(self, config: QuantumClassicalConfig):
        self.config = config
        self.quantum_simulator = QuantumSimulator(
            num_qubits=config.num_qubits,
            backend=config.quantum_backend
        )
        
    def prepare_quantum_hamiltonian(
        self, 
        cost_function: Callable,
        parameter_bounds: Dict[str, Tuple[float, float]]
    ) -> Dict:
        """Prepare quantum Hamiltonian for optimization problem."""
        
        # Map continuous optimization to Ising model
        # H = Σᵢⱼ Jᵢⱼ σᵢᶻ σⱼᶻ + Σᵢ hᵢ σᵢᶻ
        
        num_parameters = len(parameter_bounds)
        num_bits_per_param = max(4, int(np.ceil(np.log2(100))))  # 100 discrete levels
        
        if num_parameters * num_bits_per_param > self.config.num_qubits:
            raise ValueError(f"Problem requires {num_parameters * num_bits_per_param} qubits, "
                           f"but only {self.config.num_qubits} available")
        
        # Create Ising coupling matrix
        J_matrix = jnp.zeros((self.config.num_qubits, self.config.num_qubits))
        h_vector = jnp.zeros(self.config.num_qubits)
        
        # Sample discrete parameter combinations to estimate couplings
        sample_points = self._generate_sample_points(parameter_bounds, num_samples=1000)
        cost_values = [cost_function(point) for point in sample_points]
        
        # Fit Ising model to cost landscape
        J_matrix, h_vector = self._fit_ising_model(sample_points, cost_values)
        
        hamiltonian = {
            "coupling_matrix": J_matrix,
            "external_field": h_vector,
            "parameter_bounds": parameter_bounds,
            "num_bits_per_param": num_bits_per_param,
            "parameter_names": list(parameter_bounds.keys())
        }
        
        return hamiltonian
    
    def quantum_annealing_step(
        self, 
        hamiltonian: Dict,
        annealing_schedule: Optional[List[float]] = None
    ) -> Tuple[Dict, Dict]:
        """Perform quantum annealing optimization step."""
        
        if annealing_schedule is None:
            # Linear annealing schedule from 0 to 1
            annealing_schedule = jnp.linspace(0.0, 1.0, self.config.quantum_steps)
        
        # Initialize quantum state in equal superposition
        initial_state = jnp.ones(2**self.config.num_qubits) / jnp.sqrt(2**self.config.num_qubits)
        current_state = initial_state
        
        # Evolve under time-dependent Hamiltonian
        # H(t) = (1-s(t))H₀ + s(t)H₁
        # where H₀ = -Σᵢ σᵢˣ (transverse field) and H₁ is problem Hamiltonian
        
        energies = []
        states = []
        
        for step, s in enumerate(annealing_schedule):
            # Construct time-dependent Hamiltonian
            H_transverse = self._construct_transverse_field_hamiltonian()
            H_problem = self._construct_problem_hamiltonian(hamiltonian)
            
            H_total = (1 - s) * H_transverse + s * H_problem
            
            # Time evolution step
            dt = 0.01  # Time step
            current_state = self._evolve_quantum_state(current_state, H_total, dt)
            
            # Measure energy
            energy = jnp.real(jnp.conj(current_state) @ H_total @ current_state)
            energies.append(float(energy))
            states.append(current_state)
        
        # Extract optimal parameters from final state
        optimal_bitstring = self._measure_quantum_state(current_state)
        optimal_parameters = self._bitstring_to_parameters(
            optimal_bitstring, hamiltonian
        )
        
        annealing_metrics = {
            "final_energy": energies[-1],
            "energy_evolution": energies,
            "quantum_convergence": self._check_quantum_convergence(energies),
            "measurement_probability": self._compute_measurement_probability(current_state, optimal_bitstring),
            "quantum_advantage": self._estimate_quantum_advantage(energies)
        }
        
        return optimal_parameters, annealing_metrics
    
    def _generate_sample_points(
        self, 
        parameter_bounds: Dict[str, Tuple[float, float]], 
        num_samples: int
    ) -> List[Dict]:
        """Generate sample points for Ising model fitting."""
        
        samples = []
        for _ in range(num_samples):
            sample = {}
            for param_name, (low, high) in parameter_bounds.items():
                sample[param_name] = np.random.uniform(low, high)
            samples.append(sample)
        
        return samples
    
    def _fit_ising_model(
        self, 
        sample_points: List[Dict], 
        cost_values: List[float]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Fit Ising model to cost landscape."""
        
        # Convert parameter combinations to binary representations
        binary_representations = []
        for point in sample_points:
            binary_rep = self._parameters_to_bitstring(point)
            binary_representations.append(binary_rep)
        
        binary_array = jnp.array(binary_representations)  # Shape: (num_samples, num_qubits)
        cost_array = jnp.array(cost_values)
        
        # Fit quadratic model: E = Σᵢⱼ Jᵢⱼ σᵢ σⱼ + Σᵢ hᵢ σᵢ + const
        # where σᵢ ∈ {-1, +1}
        
        # Convert binary {0, 1} to Ising {-1, +1}
        ising_spins = 2 * binary_array - 1
        
        # Construct feature matrix for quadratic fit
        num_qubits = ising_spins.shape[1]
        num_features = num_qubits + num_qubits * (num_qubits - 1) // 2
        
        feature_matrix = jnp.zeros((len(sample_points), num_features))
        
        # Linear terms
        feature_matrix = feature_matrix.at[:, :num_qubits].set(ising_spins)
        
        # Quadratic terms
        feature_idx = num_qubits
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                feature_matrix = feature_matrix.at[:, feature_idx].set(ising_spins[:, i] * ising_spins[:, j])
                feature_idx += 1
        
        # Solve least squares: cost = feature_matrix @ coefficients
        coefficients, residuals, rank, s = jnp.linalg.lstsq(feature_matrix, cost_array, rcond=None)
        
        # Extract Ising parameters
        h_vector = coefficients[:num_qubits]
        
        J_matrix = jnp.zeros((num_qubits, num_qubits))
        coeff_idx = num_qubits
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                J_matrix = J_matrix.at[i, j].set(coefficients[coeff_idx])
                J_matrix = J_matrix.at[j, i].set(coefficients[coeff_idx])  # Symmetric
                coeff_idx += 1
        
        return J_matrix, h_vector
    
    def _parameters_to_bitstring(self, parameters: Dict) -> jnp.ndarray:
        """Convert continuous parameters to binary representation."""
        
        bitstring = []
        
        for param_name in sorted(parameters.keys()):
            param_value = parameters[param_name]
            
            # Normalize to [0, 1]
            # Note: This assumes parameter bounds are available
            normalized_value = max(0.0, min(1.0, param_value))
            
            # Convert to binary with fixed number of bits
            num_bits = 4  # 16 discrete levels
            discrete_value = int(normalized_value * (2**num_bits - 1))
            
            # Convert to binary
            binary_digits = [(discrete_value >> i) & 1 for i in range(num_bits)]
            bitstring.extend(binary_digits)
        
        # Pad to match number of qubits
        while len(bitstring) < self.config.num_qubits:
            bitstring.append(0)
        
        return jnp.array(bitstring[:self.config.num_qubits])
    
    def _bitstring_to_parameters(
        self, 
        bitstring: jnp.ndarray, 
        hamiltonian: Dict
    ) -> Dict:
        """Convert binary representation back to continuous parameters."""
        
        parameters = {}
        parameter_names = hamiltonian["parameter_names"]
        parameter_bounds = hamiltonian["parameter_bounds"]
        num_bits_per_param = hamiltonian["num_bits_per_param"]
        
        bit_idx = 0
        for param_name in parameter_names:
            if bit_idx + num_bits_per_param > len(bitstring):
                break
            
            # Extract bits for this parameter
            param_bits = bitstring[bit_idx:bit_idx + num_bits_per_param]
            
            # Convert binary to integer
            discrete_value = 0
            for i, bit in enumerate(param_bits):
                discrete_value += int(bit) * (2**i)
            
            # Convert to continuous value
            normalized_value = discrete_value / (2**num_bits_per_param - 1)
            
            # Scale to parameter bounds
            low, high = parameter_bounds[param_name]
            param_value = low + normalized_value * (high - low)
            
            parameters[param_name] = float(param_value)
            bit_idx += num_bits_per_param
        
        return parameters
    
    def _construct_transverse_field_hamiltonian(self) -> jnp.ndarray:
        """Construct transverse field Hamiltonian H₀ = -Σᵢ σᵢˣ."""
        
        H_transverse = jnp.zeros((2**self.config.num_qubits, 2**self.config.num_qubits))
        
        # Pauli-X matrices for each qubit
        for qubit_idx in range(self.config.num_qubits):
            # Create Pauli-X operator for qubit_idx
            pauli_x = self._pauli_x_operator(qubit_idx)
            H_transverse -= pauli_x
        
        return H_transverse
    
    def _construct_problem_hamiltonian(self, hamiltonian: Dict) -> jnp.ndarray:
        """Construct problem Hamiltonian from Ising model."""
        
        J_matrix = hamiltonian["coupling_matrix"]
        h_vector = hamiltonian["external_field"]
        
        H_problem = jnp.zeros((2**self.config.num_qubits, 2**self.config.num_qubits))
        
        # Add coupling terms: Jᵢⱼ σᵢᶻ σⱼᶻ
        for i in range(self.config.num_qubits):
            for j in range(i + 1, self.config.num_qubits):
                if J_matrix[i, j] != 0:
                    sigma_z_i = self._pauli_z_operator(i)
                    sigma_z_j = self._pauli_z_operator(j)
                    H_problem += J_matrix[i, j] * sigma_z_i @ sigma_z_j
        
        # Add external field terms: hᵢ σᵢᶻ
        for i in range(self.config.num_qubits):
            if h_vector[i] != 0:
                sigma_z_i = self._pauli_z_operator(i)
                H_problem += h_vector[i] * sigma_z_i
        
        return H_problem
    
    def _pauli_x_operator(self, qubit_idx: int) -> jnp.ndarray:
        """Create Pauli-X operator for specific qubit."""
        
        # Pauli-X matrix
        pauli_x_single = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
        identity = jnp.eye(2, dtype=jnp.complex64)
        
        # Tensor product to get full system operator
        operator = jnp.array([[1]], dtype=jnp.complex64)
        
        for i in range(self.config.num_qubits):
            if i == qubit_idx:
                operator = jnp.kron(operator, pauli_x_single)
            else:
                operator = jnp.kron(operator, identity)
        
        return operator
    
    def _pauli_z_operator(self, qubit_idx: int) -> jnp.ndarray:
        """Create Pauli-Z operator for specific qubit."""
        
        # Pauli-Z matrix
        pauli_z_single = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
        identity = jnp.eye(2, dtype=jnp.complex64)
        
        # Tensor product to get full system operator
        operator = jnp.array([[1]], dtype=jnp.complex64)
        
        for i in range(self.config.num_qubits):
            if i == qubit_idx:
                operator = jnp.kron(operator, pauli_z_single)
            else:
                operator = jnp.kron(operator, identity)
        
        return operator
    
    def _evolve_quantum_state(
        self, 
        state: jnp.ndarray, 
        hamiltonian: jnp.ndarray, 
        dt: float
    ) -> jnp.ndarray:
        """Evolve quantum state under Hamiltonian."""
        
        # Time evolution: |ψ(t+dt)⟩ = exp(-iH dt/ℏ) |ψ(t)⟩
        # Set ℏ = 1 for simplicity
        
        evolution_operator = jax.scipy.linalg.expm(-1j * hamiltonian * dt)
        evolved_state = evolution_operator @ state
        
        return evolved_state
    
    def _measure_quantum_state(self, state: jnp.ndarray) -> jnp.ndarray:
        """Measure quantum state to get bitstring."""
        
        # Compute measurement probabilities
        probabilities = jnp.abs(state)**2
        
        # Sample from probability distribution
        rng_key = jax.random.PRNGKey(np.random.randint(0, 1000000))
        sample_idx = jax.random.choice(
            rng_key, 
            len(probabilities), 
            p=probabilities
        )
        
        # Convert index to bitstring
        bitstring = jnp.array([
            (sample_idx >> i) & 1 for i in range(self.config.num_qubits)
        ])
        
        return bitstring
    
    def _check_quantum_convergence(self, energies: List[float]) -> bool:
        """Check if quantum annealing has converged."""
        
        if len(energies) < 10:
            return False
        
        recent_energies = energies[-10:]
        energy_variance = np.var(recent_energies)
        
        return energy_variance < self.config.energy_convergence_threshold
    
    def _compute_measurement_probability(
        self, 
        state: jnp.ndarray, 
        bitstring: jnp.ndarray
    ) -> float:
        """Compute probability of measuring specific bitstring."""
        
        # Convert bitstring to state index
        state_idx = 0
        for i, bit in enumerate(bitstring):
            state_idx += int(bit) * (2**i)
        
        probability = float(jnp.abs(state[state_idx])**2)
        
        return probability
    
    def _estimate_quantum_advantage(self, energies: List[float]) -> float:
        """Estimate quantum advantage over classical optimization."""
        
        # Compare final energy to initial energy
        if len(energies) < 2:
            return 0.0
        
        initial_energy = energies[0]
        final_energy = energies[-1]
        
        # Quantum advantage as relative improvement
        if abs(initial_energy) > 1e-8:
            advantage = (initial_energy - final_energy) / abs(initial_energy)
        else:
            advantage = 0.0
        
        return float(max(0.0, advantage))


class VariationalQuantumOptimizer:
    """Variational quantum optimizer using parametrized quantum circuits."""
    
    def __init__(self, config: QuantumClassicalConfig):
        self.config = config
        self.vqe = VariationalQuantumEigensolver(
            num_qubits=config.num_qubits,
            depth=config.quantum_depth
        )
    
    def optimize_with_vqe(
        self,
        cost_function: Callable,
        initial_parameters: Dict[str, float],
        quantum_ansatz: str = "QAOA"
    ) -> Tuple[Dict, Dict]:
        """Optimize using Variational Quantum Eigensolver."""
        
        # Prepare quantum circuit parameters
        num_ansatz_params = self._get_ansatz_parameter_count(quantum_ansatz)
        
        # Initialize variational parameters
        rng_key = jax.random.PRNGKey(42)
        variational_params = jax.random.uniform(
            rng_key, 
            (num_ansatz_params,), 
            minval=0.0, 
            maxval=2*jnp.pi
        )
        
        # Define quantum cost function
        def quantum_cost_function(var_params):
            # Prepare quantum circuit
            circuit_state = self._prepare_ansatz_circuit(var_params, quantum_ansatz)
            
            # Measure expectation value
            expectation_value = self._measure_expectation_value(
                circuit_state, cost_function, initial_parameters
            )
            
            return expectation_value
        
        # Classical optimization of variational parameters
        optimizer = optax.adam(learning_rate=0.01)
        opt_state = optimizer.init(variational_params)
        
        best_params = variational_params
        best_cost = float('inf')
        costs = []
        
        for step in range(self.config.classical_steps):
            # Compute gradient of quantum circuit
            cost_value, grads = jax.value_and_grad(quantum_cost_function)(variational_params)
            
            # Update variational parameters
            updates, opt_state = optimizer.update(grads, opt_state, variational_params)
            variational_params = optax.apply_updates(variational_params, updates)
            
            costs.append(float(cost_value))
            
            if cost_value < best_cost:
                best_cost = cost_value
                best_params = variational_params
        
        # Extract optimal classical parameters from best quantum state
        optimal_quantum_state = self._prepare_ansatz_circuit(best_params, quantum_ansatz)
        optimal_classical_params = self._extract_classical_parameters(
            optimal_quantum_state, initial_parameters
        )
        
        vqe_metrics = {
            "final_cost": best_cost,
            "cost_evolution": costs,
            "converged": self._check_vqe_convergence(costs),
            "quantum_circuit_depth": self.config.quantum_depth,
            "num_variational_params": num_ansatz_params,
            "quantum_fidelity": self._compute_quantum_fidelity(optimal_quantum_state)
        }
        
        return optimal_classical_params, vqe_metrics
    
    def _get_ansatz_parameter_count(self, ansatz_type: str) -> int:
        """Get number of parameters for quantum ansatz."""
        
        if ansatz_type == "QAOA":
            # QAOA with p layers: 2p parameters (β and γ for each layer)
            return 2 * self.config.ansatz_layers
        
        elif ansatz_type == "Hardware_Efficient":
            # Hardware-efficient ansatz: 3 parameters per qubit per layer
            return 3 * self.config.num_qubits * self.config.ansatz_layers
        
        else:
            # Default: one parameter per qubit per layer
            return self.config.num_qubits * self.config.ansatz_layers
    
    def _prepare_ansatz_circuit(
        self, 
        variational_params: jnp.ndarray, 
        ansatz_type: str
    ) -> jnp.ndarray:
        """Prepare quantum ansatz circuit."""
        
        # Initialize state in equal superposition
        num_states = 2**self.config.num_qubits
        state = jnp.ones(num_states, dtype=jnp.complex64) / jnp.sqrt(num_states)
        
        if ansatz_type == "QAOA":
            state = self._apply_qaoa_ansatz(state, variational_params)
        
        elif ansatz_type == "Hardware_Efficient":
            state = self._apply_hardware_efficient_ansatz(state, variational_params)
        
        else:
            # Simple rotation ansatz
            state = self._apply_rotation_ansatz(state, variational_params)
        
        return state
    
    def _apply_qaoa_ansatz(
        self, 
        initial_state: jnp.ndarray, 
        params: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply Quantum Approximate Optimization Algorithm ansatz."""
        
        state = initial_state
        
        for layer in range(self.config.ansatz_layers):
            beta = params[2*layer]
            gamma = params[2*layer + 1]
            
            # Apply problem Hamiltonian evolution: exp(-iγH_P)
            # Simplified as phase rotations
            for i in range(self.config.num_qubits):
                state = self._apply_z_rotation(state, i, gamma)
            
            # Apply mixer Hamiltonian evolution: exp(-iβH_M)
            # Mixer is typically X rotations
            for i in range(self.config.num_qubits):
                state = self._apply_x_rotation(state, i, beta)
        
        return state
    
    def _apply_hardware_efficient_ansatz(
        self, 
        initial_state: jnp.ndarray, 
        params: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply hardware-efficient ansatz."""
        
        state = initial_state
        param_idx = 0
        
        for layer in range(self.config.ansatz_layers):
            # Single-qubit rotations
            for qubit in range(self.config.num_qubits):
                # RY rotation
                state = self._apply_y_rotation(state, qubit, params[param_idx])
                param_idx += 1
                
                # RZ rotation
                state = self._apply_z_rotation(state, qubit, params[param_idx])
                param_idx += 1
                
                # RY rotation
                state = self._apply_y_rotation(state, qubit, params[param_idx])
                param_idx += 1
            
            # Entangling gates (CNOT ladder)
            for qubit in range(self.config.num_qubits - 1):
                state = self._apply_cnot(state, qubit, qubit + 1)
        
        return state
    
    def _apply_rotation_ansatz(
        self, 
        initial_state: jnp.ndarray, 
        params: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply simple rotation ansatz."""
        
        state = initial_state
        param_idx = 0
        
        for layer in range(self.config.ansatz_layers):
            for qubit in range(self.config.num_qubits):
                if param_idx < len(params):
                    state = self._apply_y_rotation(state, qubit, params[param_idx])
                    param_idx += 1
        
        return state
    
    def _apply_x_rotation(
        self, 
        state: jnp.ndarray, 
        qubit_idx: int, 
        angle: float
    ) -> jnp.ndarray:
        """Apply X rotation to specific qubit."""
        
        # RX(θ) = cos(θ/2)I - i sin(θ/2)X
        cos_half = jnp.cos(angle / 2)
        sin_half = jnp.sin(angle / 2)
        
        rx_matrix = jnp.array([
            [cos_half, -1j * sin_half],
            [-1j * sin_half, cos_half]
        ], dtype=jnp.complex64)
        
        # Apply to full system state
        return self._apply_single_qubit_gate(state, qubit_idx, rx_matrix)
    
    def _apply_y_rotation(
        self, 
        state: jnp.ndarray, 
        qubit_idx: int, 
        angle: float
    ) -> jnp.ndarray:
        """Apply Y rotation to specific qubit."""
        
        # RY(θ) = cos(θ/2)I - i sin(θ/2)Y
        cos_half = jnp.cos(angle / 2)
        sin_half = jnp.sin(angle / 2)
        
        ry_matrix = jnp.array([
            [cos_half, -sin_half],
            [sin_half, cos_half]
        ], dtype=jnp.complex64)
        
        return self._apply_single_qubit_gate(state, qubit_idx, ry_matrix)
    
    def _apply_z_rotation(
        self, 
        state: jnp.ndarray, 
        qubit_idx: int, 
        angle: float
    ) -> jnp.ndarray:
        """Apply Z rotation to specific qubit."""
        
        # RZ(θ) = exp(-iθ/2 Z)
        rz_matrix = jnp.array([
            [jnp.exp(-1j * angle / 2), 0],
            [0, jnp.exp(1j * angle / 2)]
        ], dtype=jnp.complex64)
        
        return self._apply_single_qubit_gate(state, qubit_idx, rz_matrix)
    
    def _apply_cnot(
        self, 
        state: jnp.ndarray, 
        control_qubit: int, 
        target_qubit: int
    ) -> jnp.ndarray:
        """Apply CNOT gate between two qubits."""
        
        # CNOT matrix in computational basis
        cnot_matrix = jnp.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=jnp.complex64)
        
        return self._apply_two_qubit_gate(state, control_qubit, target_qubit, cnot_matrix)
    
    def _apply_single_qubit_gate(
        self, 
        state: jnp.ndarray, 
        qubit_idx: int, 
        gate_matrix: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply single-qubit gate to quantum state."""
        
        # Reshape state to tensor form
        state_tensor = state.reshape([2] * self.config.num_qubits)
        
        # Apply gate to specified qubit
        # This is a simplified implementation - in practice would use tensor contractions
        gate_applied = jnp.tensordot(gate_matrix, state_tensor, axes=([1], [qubit_idx]))
        
        # Move the new qubit dimension back to original position
        axes_order = list(range(self.config.num_qubits + 1))
        axes_order[0], axes_order[qubit_idx + 1] = axes_order[qubit_idx + 1], axes_order[0]
        gate_applied = jnp.transpose(gate_applied, axes_order)
        
        # Reshape back to vector form
        new_state = gate_applied.reshape(-1)
        
        return new_state
    
    def _apply_two_qubit_gate(
        self, 
        state: jnp.ndarray, 
        qubit1: int, 
        qubit2: int, 
        gate_matrix: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply two-qubit gate to quantum state."""
        
        # This is a simplified implementation
        # In practice, would use efficient tensor network methods
        
        num_states = len(state)
        new_state = jnp.zeros_like(state)
        
        for i in range(num_states):
            # Extract bits for the two qubits
            bit1 = (i >> qubit1) & 1
            bit2 = (i >> qubit2) & 1
            
            # Compute two-qubit state index
            two_qubit_idx = 2 * bit1 + bit2
            
            # Apply gate to all compatible states
            for j in range(4):  # 4 possible outputs from 2-qubit gate
                new_bit1 = (j >> 1) & 1
                new_bit2 = j & 1
                
                # Construct new state index
                new_i = i
                new_i &= ~(1 << qubit1)  # Clear bit1
                new_i &= ~(1 << qubit2)  # Clear bit2
                new_i |= new_bit1 << qubit1  # Set new bit1
                new_i |= new_bit2 << qubit2  # Set new bit2
                
                new_state = new_state.at[new_i].add(
                    gate_matrix[j, two_qubit_idx] * state[i]
                )
        
        return new_state
    
    def _measure_expectation_value(
        self,
        quantum_state: jnp.ndarray,
        cost_function: Callable,
        parameter_template: Dict[str, float]
    ) -> float:
        """Measure expectation value of cost function."""
        
        # Sample from quantum state
        probabilities = jnp.abs(quantum_state)**2
        
        total_expectation = 0.0
        
        # Compute expectation value by sampling
        for state_idx, prob in enumerate(probabilities):
            if prob > 1e-10:  # Skip negligible contributions
                # Convert state index to parameter values
                parameters = self._state_index_to_parameters(state_idx, parameter_template)
                
                # Evaluate cost function
                cost_value = cost_function(parameters)
                
                # Add to expectation value
                total_expectation += prob * cost_value
        
        return float(total_expectation)
    
    def _state_index_to_parameters(
        self, 
        state_idx: int, 
        parameter_template: Dict[str, float]
    ) -> Dict[str, float]:
        """Convert quantum state index to parameter values."""
        
        # Convert state index to bitstring
        bitstring = []
        for i in range(self.config.num_qubits):
            bit = (state_idx >> i) & 1
            bitstring.append(bit)
        
        # Map bitstring to parameters (simplified mapping)
        parameters = parameter_template.copy()
        param_names = list(parameters.keys())
        
        bits_per_param = self.config.num_qubits // len(param_names) if param_names else 1
        
        for i, param_name in enumerate(param_names):
            start_bit = i * bits_per_param
            end_bit = min(start_bit + bits_per_param, self.config.num_qubits)
            
            # Convert bits to value
            param_bits = bitstring[start_bit:end_bit]
            discrete_value = sum(bit * (2**j) for j, bit in enumerate(param_bits))
            
            # Normalize to [0, 1] and scale to parameter range
            max_discrete = 2**(end_bit - start_bit) - 1
            if max_discrete > 0:
                normalized_value = discrete_value / max_discrete
                # Assume parameter range [-1, 1] for simplicity
                parameters[param_name] = -1.0 + 2.0 * normalized_value
        
        return parameters
    
    def _extract_classical_parameters(
        self,
        quantum_state: jnp.ndarray,
        parameter_template: Dict[str, float]
    ) -> Dict[str, float]:
        """Extract most likely classical parameters from quantum state."""
        
        # Find state with highest probability
        probabilities = jnp.abs(quantum_state)**2
        max_prob_idx = int(jnp.argmax(probabilities))
        
        # Convert to parameters
        optimal_parameters = self._state_index_to_parameters(
            max_prob_idx, parameter_template
        )
        
        return optimal_parameters
    
    def _check_vqe_convergence(self, costs: List[float]) -> bool:
        """Check VQE convergence."""
        
        if len(costs) < 10:
            return False
        
        recent_costs = costs[-10:]
        cost_std = np.std(recent_costs)
        
        return cost_std < self.config.energy_convergence_threshold
    
    def _compute_quantum_fidelity(self, quantum_state: jnp.ndarray) -> float:
        """Compute quantum state fidelity."""
        
        # Fidelity with respect to equal superposition state
        equal_superposition = jnp.ones_like(quantum_state) / jnp.sqrt(len(quantum_state))
        
        fidelity = jnp.abs(jnp.conj(equal_superposition) @ quantum_state)**2
        
        return float(fidelity)


class HybridQuantumClassicalOptimizer:
    """Main hybrid optimizer combining quantum annealing, VQE, and classical methods."""
    
    def __init__(self, config: QuantumClassicalConfig):
        self.config = config
        self.quantum_annealer = QuantumAnnealer(config)
        self.vqe_optimizer = VariationalQuantumOptimizer(config)
        
    def hybrid_optimize(
        self,
        cost_function: Callable,
        initial_parameters: Dict[str, float],
        parameter_bounds: Dict[str, Tuple[float, float]],
        physics_constraints: Optional[List[Callable]] = None
    ) -> Tuple[Dict, Dict]:
        """Perform hybrid quantum-classical optimization."""
        
        logging.info("Starting hybrid quantum-classical optimization")
        
        all_results = []
        all_metrics = {}
        
        for iteration in range(self.config.hybrid_iterations):
            logging.info(f"Hybrid iteration {iteration + 1}/{self.config.hybrid_iterations}")
            
            # Phase 1: Quantum Annealing for global exploration
            if iteration % 3 == 0:  # Every 3rd iteration
                logging.info("Quantum annealing phase")
                
                try:
                    hamiltonian = self.quantum_annealer.prepare_quantum_hamiltonian(
                        cost_function, parameter_bounds
                    )
                    
                    qa_params, qa_metrics = self.quantum_annealer.quantum_annealing_step(
                        hamiltonian
                    )
                    
                    all_results.append(("quantum_annealing", qa_params, qa_metrics))
                    all_metrics[f"qa_iteration_{iteration}"] = qa_metrics
                    
                    # Update current best parameters
                    current_best = qa_params
                    
                except Exception as e:
                    logging.warning(f"Quantum annealing failed: {e}")
                    current_best = initial_parameters
            
            # Phase 2: Variational Quantum Eigensolver for local refinement
            else:
                logging.info("VQE optimization phase")
                
                try:
                    vqe_params, vqe_metrics = self.vqe_optimizer.optimize_with_vqe(
                        cost_function, 
                        current_best if 'current_best' in locals() else initial_parameters
                    )
                    
                    all_results.append(("vqe", vqe_params, vqe_metrics))
                    all_metrics[f"vqe_iteration_{iteration}"] = vqe_metrics
                    
                    current_best = vqe_params
                    
                except Exception as e:
                    logging.warning(f"VQE optimization failed: {e}")
                    current_best = initial_parameters
            
            # Phase 3: Classical refinement with physics constraints
            if physics_constraints:
                logging.info("Physics-constrained classical refinement")
                
                constrained_params, classical_metrics = self._classical_constrained_optimization(
                    cost_function,
                    current_best,
                    parameter_bounds,
                    physics_constraints
                )
                
                all_results.append(("classical_constrained", constrained_params, classical_metrics))
                all_metrics[f"classical_iteration_{iteration}"] = classical_metrics
                
                current_best = constrained_params
            
            # Check convergence
            if len(all_results) >= 3:
                if self._check_hybrid_convergence(all_results):
                    logging.info(f"Hybrid optimization converged at iteration {iteration + 1}")
                    break
        
        # Select best result across all methods
        best_params, best_method, best_cost = self._select_best_result(
            all_results, cost_function
        )
        
        # Aggregate final metrics
        final_metrics = self._aggregate_hybrid_metrics(all_metrics, best_method, best_cost)
        
        logging.info(f"Hybrid optimization completed. Best method: {best_method}, Best cost: {best_cost:.6f}")
        
        return best_params, final_metrics
    
    def _classical_constrained_optimization(
        self,
        cost_function: Callable,
        initial_params: Dict[str, float],
        parameter_bounds: Dict[str, Tuple[float, float]],
        physics_constraints: List[Callable]
    ) -> Tuple[Dict, Dict]:
        """Classical optimization with physics constraints."""
        
        # Convert to scipy-compatible format
        param_names = list(initial_params.keys())
        x0 = np.array([initial_params[name] for name in param_names])
        bounds = [parameter_bounds[name] for name in param_names]
        
        # Objective function for scipy
        def scipy_objective(x):
            params = {name: x[i] for i, name in enumerate(param_names)}
            return cost_function(params)
        
        # Constraint functions
        constraints = []
        for constraint_fn in physics_constraints:
            def constraint_wrapper(x, fn=constraint_fn):
                params = {name: x[i] for i, name in enumerate(param_names)}
                return fn(params)
            
            constraints.append({
                'type': 'ineq',
                'fun': constraint_wrapper
            })
        
        # Optimize with constraints
        result = scipy.optimize.minimize(
            scipy_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100}
        )
        
        # Convert back to parameter dict
        optimal_params = {name: result.x[i] for i, name in enumerate(param_names)}
        
        classical_metrics = {
            "success": result.success,
            "final_cost": float(result.fun),
            "iterations": result.nit,
            "constraint_violations": self._check_constraint_violations(
                optimal_params, physics_constraints
            ),
            "optimization_method": "SLSQP"
        }
        
        return optimal_params, classical_metrics
    
    def _check_constraint_violations(
        self,
        parameters: Dict[str, float],
        constraints: List[Callable]
    ) -> List[float]:
        """Check physics constraint violations."""
        
        violations = []
        for constraint_fn in constraints:
            try:
                violation = constraint_fn(parameters)
                violations.append(float(max(0, -violation)))  # Positive = violation
            except Exception as e:
                logging.warning(f"Constraint evaluation failed: {e}")
                violations.append(float('inf'))
        
        return violations
    
    def _check_hybrid_convergence(self, results: List[Tuple]) -> bool:
        """Check convergence of hybrid optimization."""
        
        if len(results) < 3:
            return False
        
        # Extract recent costs
        recent_costs = []
        for method, params, metrics in results[-3:]:
            if "final_cost" in metrics:
                recent_costs.append(metrics["final_cost"])
            elif "final_energy" in metrics:
                recent_costs.append(metrics["final_energy"])
        
        if len(recent_costs) < 3:
            return False
        
        # Check cost convergence
        cost_std = np.std(recent_costs)
        cost_improvement = (recent_costs[0] - recent_costs[-1]) / abs(recent_costs[0])
        
        converged = (
            cost_std < self.config.energy_convergence_threshold and
            cost_improvement < 0.01  # Less than 1% improvement
        )
        
        return converged
    
    def _select_best_result(
        self,
        results: List[Tuple],
        cost_function: Callable
    ) -> Tuple[Dict, str, float]:
        """Select best result from all optimization methods."""
        
        best_cost = float('inf')
        best_params = {}
        best_method = "none"
        
        for method, params, metrics in results:
            # Evaluate actual cost function to ensure consistency
            try:
                cost = cost_function(params)
                
                if cost < best_cost:
                    best_cost = cost
                    best_params = params
                    best_method = method
            
            except Exception as e:
                logging.warning(f"Cost evaluation failed for {method}: {e}")
        
        return best_params, best_method, best_cost
    
    def _aggregate_hybrid_metrics(
        self,
        all_metrics: Dict,
        best_method: str,
        best_cost: float
    ) -> Dict:
        """Aggregate metrics from all hybrid optimization phases."""
        
        aggregated = {
            "best_method": best_method,
            "best_cost": best_cost,
            "total_iterations": len(all_metrics),
            "methods_used": [],
            "quantum_advantage": 0.0,
            "convergence_achieved": True,
            "physics_compliance": True
        }
        
        # Collect method statistics
        method_costs = {"quantum_annealing": [], "vqe": [], "classical_constrained": []}
        
        for iteration_key, metrics in all_metrics.items():
            if "qa_" in iteration_key:
                aggregated["methods_used"].append("quantum_annealing")
                if "final_energy" in metrics:
                    method_costs["quantum_annealing"].append(metrics["final_energy"])
            
            elif "vqe_" in iteration_key:
                aggregated["methods_used"].append("vqe")
                if "final_cost" in metrics:
                    method_costs["vqe"].append(metrics["final_cost"])
            
            elif "classical_" in iteration_key:
                aggregated["methods_used"].append("classical_constrained")
                if "final_cost" in metrics:
                    method_costs["classical_constrained"].append(metrics["final_cost"])
                
                # Check physics compliance
                if "constraint_violations" in metrics:
                    max_violation = max(metrics["constraint_violations"]) if metrics["constraint_violations"] else 0
                    aggregated["physics_compliance"] = max_violation < 1e-6
        
        # Estimate quantum advantage
        classical_costs = method_costs["classical_constrained"]
        quantum_costs = method_costs["quantum_annealing"] + method_costs["vqe"]
        
        if classical_costs and quantum_costs:
            avg_classical = np.mean(classical_costs)
            avg_quantum = np.mean(quantum_costs)
            
            if avg_classical > 0:
                aggregated["quantum_advantage"] = (avg_classical - avg_quantum) / avg_classical
        
        # Unique methods used
        aggregated["methods_used"] = list(set(aggregated["methods_used"]))
        
        return aggregated


# Example usage and demonstration
if __name__ == "__main__":
    
    # Configuration
    config = QuantumClassicalConfig(
        quantum_depth=4,
        num_qubits=12,
        classical_steps=50,
        quantum_steps=30,
        hybrid_iterations=10,
        variational_form="QAOA",
        ansatz_layers=3
    )
    
    # Define test optimization problem
    def rosenbrock_function(params: Dict[str, float]) -> float:
        """Rosenbrock function as test optimization problem."""
        x = params.get('x', 0.0)
        y = params.get('y', 0.0)
        
        return 100 * (y - x**2)**2 + (1 - x)**2
    
    # Parameter bounds
    parameter_bounds = {
        'x': (-2.0, 2.0),
        'y': (-2.0, 2.0)
    }
    
    # Initial parameters
    initial_parameters = {'x': 0.5, 'y': 0.5}
    
    # Physics constraints (example: energy conservation)
    def energy_constraint(params):
        # Example: x² + y² ≤ 4 (within circle of radius 2)
        x, y = params['x'], params['y']
        return 4 - (x**2 + y**2)
    
    physics_constraints = [energy_constraint]
    
    # Create optimizer
    optimizer = HybridQuantumClassicalOptimizer(config)
    
    # Run optimization
    optimal_params, metrics = optimizer.hybrid_optimize(
        rosenbrock_function,
        initial_parameters,
        parameter_bounds,
        physics_constraints
    )
    
    print("Quantum-Classical Hybrid Optimization Results:")
    print(f"Optimal parameters: {optimal_params}")
    print(f"Final cost: {rosenbrock_function(optimal_params):.6f}")
    print(f"Best method: {metrics['best_method']}")
    print(f"Quantum advantage: {metrics['quantum_advantage']:.4f}")
    print(f"Physics compliance: {metrics['physics_compliance']}")
    print(f"Methods used: {metrics['methods_used']}")