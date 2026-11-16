# Quantum Time-Series Transformer (PennyLane Implementation)
This document describes the PyTorch and PennyLane implementation of the quantum time-series transformer developed by Park et al. (2025), as used in this repository (`qtst/qts_transformer.py`, and inline in `notebooks/QTST_ABIDE_demo.ipynb`). The quantum time-series transformer is based on the Quixer model architecture proposed by Khatri et al. (2024). This model is designed to process time-series data by leveraging quantum computing principles for embedding, mixing, and non-linear transformation of temporal features.

# Core Concepts
The model's architecture can be broken down into four key stages:

- **Unitary Temporal Embedding**: The feature vector at each point in a time sequence is mapped to a unique quantum circuit (a unitary matrix), creating a quantum representation of that specific moment's data.

- **Time Sequence Mixing (LCU)**: A Linear Combination of Unitaries (LCU) is used to create a weighted superposition ($M$) of all the time-step unitaries ($U_j$). This serves as a quantum-native attention-like mechanism to mix information across the time sequence. In the code, each weight $e^{i\gamma_j}{|a_j|}^2$ is one learnable complex number.

$$M = \sum^{n-1}_{j=0}e^{i\gamma_j}{|a_j|}^2U_j$$

- **Non-linearity (QSVT)**: A Quantum Singular Value Transform (QSVT) is simulated to apply a non-linear polynomial transformation to the mixed state ($M$). This allows the model to capture richer, higher-order interactions between different time-steps.

$$P_c(M) = c_d(M^d) + c_{d-1}(M^{d-1}) + \cdots + c_1M+c_0I$$

- **Readout and Classical Processing**: The final quantum state is measured to extract classical features (Pauli expectation values). These features are then processed by a classical feed-forward neural network to produce the final prediction.

# Code Implementation Breakdown
This section explains how each conceptual step is realized in the `qtst/qts_transformer.py` code.

## 1) Unitary Temporal Embedding
**Concept**: Classical features for each time-step are mapped to a unique Parameterized Quantum Circuit (PQC), which defines a unitary transformation.

### Realization in the Code:

This process involves three main parts:

**1-1) Classical Projection (forward method)**

The classical input feature vector for each time-step is passed through a standard torch.nn.Linear layer to generate a set of rotation angles for the quantum gates.
```
# QuantumTSTransformer -> forward()
# Input x has shape (batch, n_timesteps, feature_dim)
# The linear layer maps feature_dim -> n_rots (number of rotation angles);
# the sigmoid keeps the angles in a consistent range
timestep_params = self.rot_sigm(self.feature_projection(self.dropout(x)))
```
- self.feature_projection: The learnable linear layer.
- timestep_params: The resulting tensor holding the gate parameters for every time-step in the batch.

**1-2) Quantum Circuit Definition (sim14_circuit function)**

This function defines the fixed structure of the PQC, arranging RY and CRX gates. When parameterized by timestep_params, it defines a unique unitary matrix U for that time-step.
```
def sim14_circuit(params, wires, layers=1):
    # ... circuit logic with qml.RY and qml.CRX gates ...
```

**1-3) QNode for State Evolution (__init__ method)**

The `_timestep_state` QNode (stored as `self.timestep_state_qnode`) is the PennyLane object that simulates the quantum evolution. It takes an initial state and the timestep_params and applies the unitary transformation defined by sim14_circuit.
```
# QuantumTSTransformer -> __init__()
@qml.qnode(...)
def _timestep_state(initial_state, params):
    qml.StatePrep(initial_state, wires=range(n_wires))
    sim14_circuit(params, wires=n_wires, layers=n_layers)
    return qml.state()
```
## 2) Time Sequence Mixing via LCU
**Concept**: The model learns a complex-valued weight for each time-step and creates a weighted sum of the quantum states produced by each time-step's unitary.

### Realization in the Code:

**2-1) Trainable Coefficients (__init__ method)**

The complex weights for the LCU are defined as a learnable model parameter.
```
# QuantumTSTransformer -> __init__()
self.mix_coeffs = torch.nn.Parameter(torch.rand(n_timesteps, dtype=torch.complex64))
```
**2-2) LCU Application (apply_lcu_mixing function)**

This function simulates the LCU. It first calculates the evolved state for every time-step's unitary (U_t |ψ⟩) in a vectorized manner and then performs the weighted sum using torch.einsum.
```
# apply_lcu_mixing()
# ... calculates evolved (shape: batch, time, amplitudes) ...

# This is the LCU:
# It multiplies each evolved state vector by its learned complex coefficient
# and sums them up, creating the superposition.
return torch.einsum("bti,bt->bi", evolved, coeffs.to(torch.complex64))
```

## 3) Non-linearity via QSVT

**Concept**: A polynomial transformation P(M) is applied to the LCU superposition M. This is simulated by iteratively applying the LCU operation and summing the results according to learnable polynomial coefficients.

### Realization in the Code:

**3-1) Trainable Coefficients (__init__ method)**

The coefficients of the polynomial (c₀, c₁, c₂, ...) are defined as a learnable parameter.
```
# QuantumTSTransformer -> __init__()
self.poly_coeffs = torch.nn.Parameter(torch.rand(degree + 1))
```
**3-2) Polynomial Application (apply_qsvt_polynomial function)**

This function orchestrates the QSVT simulation by looping through the polynomial coefficients and repeatedly calling apply_lcu_mixing to compute the higher-order terms (M|ψ⟩, M(M|ψ⟩), etc.). The result is divided by the sum of the absolute values of the coefficients (their L1 norm).
```
# apply_qsvt_polynomial()

# Zeroth-order term (c₀ * I)|ψ⟩
acc = poly_coeffs[0] * base_states
working = base_states

# Loop for higher-order terms
for c in poly_coeffs[1:]:
    # Calculate the next power, e.g., M(working)
    working = apply_lcu_mixing(working, ...)
    # Add the weighted term to the accumulator: acc += c_k * M^k |ψ⟩
    acc = acc + c * working
return acc / torch.linalg.vector_norm(poly_coeffs, ord=1)
```

## 4) Readout and Classical Processing
**Concept**: The final quantum state is measured to extract a classical feature vector. This vector is then processed by a classical feed-forward network to produce the final output.

### Realization in the Code:

**4-1) Measurement QNode (__init__ method)**

The `_qff_expval` QNode (stored as `self.qff_qnode_expval`) performs the measurement. It applies a final trainable PQC and then computes the expectation values of Pauli X, Y, and Z operators for each qubit.
```
# QuantumTSTransformer -> __init__()

@qml.qnode(...)
def _qff_expval(initial_state, params):
    qml.StatePrep(initial_state, wires=range(n_wires))
    sim14_circuit(params, wires=n_wires, layers=1)  # Final trainable quantum layer
    # Define the Pauli measurements
    obs = [qml.PauliX(i) for i in range(n_wires)] + \
          [qml.PauliY(i) for i in range(n_wires)] + \
          [qml.PauliZ(i) for i in range(n_wires)]
    return [qml.expval(o) for o in obs]
```
**4-2) Execution and Classical Layer (forward method)**

The forward pass normalizes the mixed state, executes the measurement and passes the classical results to the final linear layer.
```
# QuantumTSTransformer -> forward()

# Normalize the state produced by the QSVT polynomial
norm = torch.linalg.vector_norm(mixed, dim=1, keepdim=True)
mixed = mixed / (norm + 1e-9)

# Call the measurement QNode to get classical expectation values,
# stack them and feed them into the final linear layer
exps = torch.stack(self.qff_qnode_expval(initial_state=mixed, params=self.qff_params), dim=1)
out = self.output_ff(exps.float())
return out.squeeze(-1)
```
- self.output_ff: The final torch.nn.Linear layer that maps the 3 * n_qubits measurement outcomes to the desired output_dim.

# Reference
- Park, J. J., Seo, J., Bae, S., Chen, S. Y.-C., Tseng, H.-H., Cha, J., & Yoo, S. (2025, Aug 31 - Sep 5). Resting-state fMRI Analysis using Quantum Time-series Transformer. _2025 IEEE International Conference on Quantum Computing and Engineering (QCE)_, Albuquerque, New Mexico, USA. https://doi.org/10.48550/arXiv.2509.00711
- Khatri, N., Matos, G., Coopmans, L., & Clark, S. (2024). Quixer: A Quantum Transformer Model. _arXiv:2406.04305_.
