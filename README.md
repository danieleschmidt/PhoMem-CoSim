# PhoMem-CoSim

**Photonic-Memristive Co-Simulator** — a Python package for simulating
integrated photonic circuits with PCM memristive weights.

Models an MZI (Mach-Zehnder Interferometer) mesh optical layer coupled to
Phase Change Material (PCM) memristive weights, enabling exploration of
photonic neuromorphic computing with realistic device physics.

---

## Features

| Component | Description |
|---|---|
| `OpticalLayer` | N-mode MZI mesh implementing a parameterised unitary (Clements scheme) |
| `MemristiveWeight` | GST PCM cell with amorphous/partial/crystalline state machine |
| `CoDifferentiableSim` | Combined forward pass + finite-difference gradient engine |
| `EnergyModel` | Optical and memristive energy accounting |

---

## Installation

```bash
pip install numpy pytest
# No other dependencies required
```

---

## Quick Start

```python
import numpy as np
from phomem_cosim import OpticalLayer, MemristiveWeight, CoDifferentiableSim, EnergyModel

# 4-mode MZI mesh
layer = OpticalLayer(n_modes=4, seed=42)

# Input field
x = np.array([1, 0, 0, 0], dtype=complex)
y = layer.forward(x)
print(f"Output: {y}")
print(f"Norm preserved: {np.isclose(np.linalg.norm(x), np.linalg.norm(y))}")

# PCM memristive weight
cell = MemristiveWeight()
print(cell.get_state())          # "amorphous"
cell.set_pulse(5)
print(cell.get_state())          # "partial"
print(f"G = {cell.get_conductance():.3e} S")
cell.set_pulse(10)
print(cell.get_state())          # "crystalline"
cell.reset_pulse()
print(cell.get_state())          # "amorphous"
```

---

## Run the Demo

```bash
python demo.py
```

This creates a 4-mode photonic network, programs memristive cells with
different numbers of SET pulses, runs a forward pass, computes a toy
gradient, and reports energy breakdown.

---

## Run Tests

```bash
python -m pytest tests/ -v
```

---

## Physics Background

### MZI Mesh (Clements Decomposition)

The optical layer implements a universal N-mode unitary via a triangular
mesh of Mach-Zehnder Interferometers. Each MZI is parameterised by:

- **θ** (theta): splitting ratio — controls power coupling
- **φ** (phi): phase shift — controls interference

Transfer matrix for a single MZI:

```
U_ij = [[ e^{iφ}·cos(θ/2),  -sin(θ/2) ],
         [ e^{iφ}·sin(θ/2),   cos(θ/2) ]]
```

The full unitary is the product of all MZI matrices in column order.

### PCM Memristive Weight

Germanium-Antimony-Telluride (GST) Phase Change Material cells:

| State | Crystallization | Resistance | Conductance |
|---|---|---|---|
| Amorphous | f = 0.0 | ~1 MΩ | ~1 µS |
| Partial | 0 < f < 1 | intermediate | intermediate |
| Crystalline | f = 1.0 | ~1 kΩ | ~1 mS |

Conductance follows log-linear interpolation:
```
G(f) = G_amo × (G_cryst / G_amo)^f
```

### Energy Model

| Source | Formula |
|---|---|
| Optical | `E = dt × P_mode × Σ|x_i|²` |
| Memristive | `E = N_pulses × E_pulse` |

Typical values: dt = 1 ns, P_mode = 1 mW, E_pulse = 1 pJ.

---

## Architecture

```
phomem_cosim/
├── __init__.py              # Package exports
├── optical_layer.py         # OpticalLayer (MZI mesh)
├── memristive_weight.py     # MemristiveWeight (PCM)
├── co_differentiable_sim.py # CoDifferentiableSim
└── energy_model.py          # EnergyModel

tests/
└── test_phomem_cosim.py     # 13 unit + integration tests

demo.py                      # End-to-end demonstration
```

---

## License

MIT
