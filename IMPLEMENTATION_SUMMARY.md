# Implementation Summary: Final Measurements and Logical Operators

## Overview

This document summarizes the implementation of final data qubit measurements and logical operator computation for quantum error correction circuits. The implementation uses the `galois` library for binary field algebra to automatically compute logical operators from stabilizer generators.

## Key Components

### 1. LogicalOperators Class (`logical_operators.py`)

**Purpose**: Computes logical X and Z operators using binary field algebra.

**Key Features**:
- Constructs parity check matrices `hx` and `hz` from stabilizer supports
- Finds logical operators in the quotient space: nullspace(h) / rowspace(h')
- Returns minimal independent set of logical operators
- Provides utility methods for querying operator properties

**Algorithm**:
1. Build parity check matrices where:
   - Each row = one stabilizer (X or Z)
   - Each column = one data qubit (ordered by lattice point, then L/R)
2. For logical Z operators:
   - Find nullspace of `hx` (vectors orthogonal to X stabilizers)
   - Remove vectors in rowspace of `hz` (Z stabilizer space)
3. For logical X operators:
   - Find nullspace of `hz` (vectors orthogonal to Z stabilizers)
   - Remove vectors in rowspace of `hx` (X stabilizer space)
4. Select minimal independent representatives

**Example Usage**:
```python
from logical_operators import LogicalOperators

log_ops = LogicalOperators(qubit_system, lattice_points)
log_ops.print_summary()

# Access operators
num_logical = log_ops.get_num_logical_qubits()
z_ops = log_ops.logical_z_ops  # List of binary vectors
x_ops = log_ops.logical_x_ops
```

### 2. Final Measurements (`syndrome_circuit.py`)

**Purpose**: Add final measurements of all data qubits and optional initial state preparation.

**Key Features**:
- Supports both Z-basis and X-basis measurements (configurable via `basis` parameter)
- X-basis mode adds initial `RX` resets to prepare |+⟩ states
- Data qubits measured at end in consistent order (by lattice point, then L/R)
- Measurement indices tracked for observable construction

**Parameters**:
- `basis='Z'`: Measure data qubits in Z-basis (default)
- `basis='X'`: Prepare with RX, measure in X-basis

### 3. Observable Instructions (`circuit_operations.py`)

**Purpose**: Represent Stim `OBSERVABLE_INCLUDE` instructions.

**Key Features**:
- Stores numeric observable ID and measurement record references
- Generates Stim instruction: `OBSERVABLE_INCLUDE(id) rec[-k1] rec[-k2] ...`
- Integrated into time-ordered operation framework

**Example**:
```python
# Observable for logical operator with support on qubits 0, 4
obs = Observable(time=10.0, observable_id=0, measurement_indices=[8, 4])
print(obs.to_stim())  # "OBSERVABLE_INCLUDE(0) rec[-8] rec[-4]"
```

### 4. Integration into SyndromeCircuit

**New Constructor Parameters**:
```python
SyndromeCircuit(
    ...,
    basis='Z',                    # Measurement basis for data qubits
    include_observables=True      # Whether to compute and include logical operators
)
```

**Build Process**:
1. (Optional) Add initial state preparation if `basis='X'`
2. Run syndrome extraction cycles
3. Add final data qubit measurements
4. (Optional) Compute and add observable instructions

**Logical Operator Construction**:
- Automatically computed during initialization if `include_observables=True`
- Uses binary field algebra (galois library)
- Observables added at end of circuit referencing final measurements
- Observable IDs: 0, 1, 2, ... for each logical qubit

## Binary Field Algebra with Galois

The `galois` library provides:
- GF(2) finite field arithmetic
- Nullspace computation: `matrix.null_space()`
- Rowspace computation: `matrix.row_space()`
- Efficient linear algebra over binary fields

**Example**:
```python
import galois
GF = galois.GF(2)

# Create parity check matrix
hx = GF([[1, 1, 0, 1],
         [0, 1, 1, 1]])

# Find nullspace (logical operators orthogonal to stabilizers)
nullspace = hx.null_space()
```

## Data Qubit Ordering

Data qubits are ordered consistently for parity check matrices and observables:

```
For lattice points [p0, p1, p2, ...]:
  Column 0: L qubit at p0
  Column 1: R qubit at p0
  Column 2: L qubit at p1
  Column 3: R qubit at p1
  ...
```

This ordering ensures:
- Consistent parity check matrix construction
- Correct measurement index references in observables
- Easy mapping between logical operators and physical qubits

## Testing and Examples

### Example: `example_final_measurements_and_observables()`

Demonstrates:
- Z-basis measurement mode
- X-basis measurement mode with initial RX
- Logical operator computation
- Observable inclusion in Stim circuits
- Circuit comparison between modes

**Output** (excerpt):
```
Code parameters:
  Number of data qubits: 8
  Number of X stabilizers: 4
  Number of Z stabilizers: 4
  Number of logical X operators found: 3
  Number of logical Z operators found: 3
  Number of logical qubits: 3

Logical Z operators:
  ZL0: weight=2, qubits=[0, 4]
  ZL1: weight=4, qubits=[1, 4, 6, 7]
  ZL2: weight=2, qubits=[2, 6]

Stim circuit (last lines):
  M 0 1 2 3 4 5 6 7
  OBSERVABLE_INCLUDE(0) rec[-8] rec[-4]
  OBSERVABLE_INCLUDE(1) rec[-7] rec[-4] rec[-2] rec[-1]
  OBSERVABLE_INCLUDE(2) rec[-6] rec[-2]
```

## Backwards Compatibility

All existing functionality preserved:
- Circuits without observables work as before
- Old constructor signatures supported (e.g., `num_cycles` parameter)
- Examples and tests remain functional
- No breaking changes to existing APIs

## Performance Considerations

- Logical operator computation is O(n³) for n data qubits (Gaussian elimination)
- For large codes (>100 data qubits), computation may take seconds
- Computation done once during circuit initialization
- Can disable with `include_observables=False` if not needed

## Future Extensions

Potential improvements:
1. **Optimized Logical Operators**: Select low-weight representatives
2. **Logical State Initialization**: Add preparation of logical |0⟩ or |+⟩
3. **Fault-Tolerant Measurements**: Multi-round final measurements with error correction
4. **Custom Observable Selection**: Allow user to specify which logical operators to include
5. **Syndrome-based Observables**: Combine with syndrome history for better error detection

## Dependencies

New dependency added:
```
galois>=0.3.0
```

Provides:
- Binary field (GF(2)) arithmetic
- Nullspace and rowspace computation
- Efficient linear algebra for error correction codes

## File Changes Summary

1. **requirements.txt**: Added `galois>=0.3.0`
2. **circuit_operations.py**: Added `Observable` class
3. **logical_operators.py**: New file, implements logical operator computation
4. **syndrome_circuit.py**: 
   - Added `basis` and `include_observables` parameters
   - Added methods: `_add_initial_state_preparation`, `_add_final_measurements`, `_add_observables`
   - Integrated logical operator computation
5. **examples.py**: Added `example_final_measurements_and_observables()`
6. **README.md**: Updated with new features and examples

## Usage Recommendations

**For Z-basis memory experiments**:
```python
circuit = SyndromeCircuit(
    qubit_system, lattice_points, gate_order,
    num_cycles=1,
    basis='Z',
    include_observables=True
)
```

**For X-basis memory experiments**:
```python
circuit = SyndromeCircuit(
    qubit_system, lattice_points, gate_order,
    num_cycles=1,
    basis='X',
    include_observables=True
)
```

**For noisy simulations with error decoding**:
```python
circuit = SyndromeCircuit(
    qubit_system, lattice_points, gate_order,
    num_noisy_cycles=10,
    p_cx=0.001,
    basis='Z',
    include_observables=True
)

stim_circuit = circuit.to_stim_circuit()
# Use stim_circuit with sampler for Monte Carlo simulations
```

