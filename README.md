# Lattice Qubit Framework for Bivariate Bicycle Quantum Code

This framework provides tools for working with D-dimensional lattices with periodic boundary conditions, designed for simulating the bivariate bicycle quantum code with time vortex defects. Each lattice point hosts four qubits: two data qubits (L, R) and two ancilla qubits (X_anc, Z_anc) for stabilizer measurements.

## Features

- **LatticePoint**: Immutable representation of points in Z^D with arithmetic operations
- **Lattice**: D-dimensional lattice with periodic boundary conditions defined by lattice vectors
- **QubitSystem**: Management of qubits on lattice points with Stim circuit integration
- **SyndromeCircuit**: Time-ordered syndrome extraction circuit builder with noise and detector support
- **LogicalOperators**: Automatic computation of logical operators using binary field algebra (Galois library)
- Support for arbitrary lattice vectors (not just orthogonal)
- Efficient point normalization and equivalence checking
- Ready for Stim circuit generation with observables for error correction

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from lattice import Lattice, LatticePoint
from qubit_system import QubitSystem

# Create a 2D lattice with 4x4 periodic boundary conditions
lattice_vectors = [[4, 0], [0, 4]]
lattice = Lattice(lattice_vectors)

# Create qubit system
qubit_system = QubitSystem(lattice)

# Access qubits at a lattice point
point = LatticePoint([1, 1])
qubits = qubit_system.get_all_qubits_at_point(point)
print(qubits)  # {'L': 0, 'R': 1, 'X_anc': 2, 'Z_anc': 3}

# Get qubit indices for CX gates (X_anc at point -> L at shifted point)
qubit1, qubit2 = qubit_system.get_axis_shifted_pair(point, 'X_anc', 0, 'L')
print(f"CX {qubit1} {qubit2}")  # CX 2 4
```

## Examples

Run the examples to see the framework in action:

```bash
python examples.py
```

This demonstrates:
- 2D toric code lattice
- 4D hypercubic lattice  
- Custom non-orthogonal lattices
- Stim circuit generation
- Syndrome extraction circuits with custom gate orders
- Noisy circuits with depolarizing noise
- Detectors for stabilizer measurements
- Final data qubit measurements and logical observables

## API Reference

### LatticePoint

Immutable point in Z^D with coordinates stored as NumPy array.

```python
point = LatticePoint([1, 2, 3])
point2 = point + [1, 0, 0]  # LatticePoint([2, 2, 3])
point3 = 2 * point          # LatticePoint([2, 4, 6])
```

### Lattice

D-dimensional lattice with periodic boundary conditions.

```python
# Create lattice from vectors
lattice = Lattice([[4, 0], [0, 4]])  # 2D 4x4 lattice

# Normalize points to canonical form
canonical = lattice.normalize_point([5, 1])  # Returns LatticePoint([1, 1])

# Check point equivalence
lattice.are_equivalent([1, 1], [5, 1])  # True

# Get shifted points
shifted = lattice.get_shifted_point([1, 1], [1, 0])  # LatticePoint([2, 1])
```

### QubitSystem

Manages qubits on lattice points for circuit generation.

```python
qubit_system = QubitSystem(lattice)

# Get qubit index
index = qubit_system.get_qubit_index([1, 1], 'L')

# Get qubit pairs for gates
q1, q2 = qubit_system.get_axis_shifted_pair([1, 1], 'X_anc', 0, 'L')

# Get all qubits at a point
all_qubits = qubit_system.get_all_qubits_at_point([1, 1])
```

### SyndromeCircuit

Builds complete syndrome extraction circuits with time-ordered operations.

```python
from syndrome_circuit import SyndromeCircuit
from gate_order import GateOrder

# Get all lattice points
lattice_points = lattice.get_all_lattice_points()

# Create default gate order
gate_order = GateOrder.get_default_order(lattice.dimension)

# Build circuit with final measurements and observables
circuit = SyndromeCircuit(
    qubit_system, 
    lattice_points, 
    gate_order,
    num_cycles=3,
    basis='Z',  # or 'X' for X-basis measurements
    include_observables=True  # Include logical operators
)

# Generate Stim circuit
stim_circuit = circuit.to_stim_circuit()
print(stim_circuit)
```

### LogicalOperators

Computes logical X and Z operators using binary field algebra.

```python
from logical_operators import LogicalOperators

# Compute logical operators
log_ops = LogicalOperators(qubit_system, lattice_points)

# Print summary
log_ops.print_summary()

# Access operators
num_logical_qubits = log_ops.get_num_logical_qubits()
logical_x_ops = log_ops.logical_x_ops  # List of binary vectors
logical_z_ops = log_ops.logical_z_ops  # List of binary vectors

# Get qubits in an operator's support
qubits = log_ops.get_observable_qubits(logical_z_ops[0])
```

## Advanced Features

### Custom Gate Orders

Specify the exact order of CX gates in syndrome extraction:

```python
from gate_order import GateDescriptor, GateOrder

# Define custom order
descriptors = [
    GateDescriptor("Z", "on_site_L"),
    GateDescriptor("Z", "on_site_R"),
    GateDescriptor("X", "axis_0"),
    GateDescriptor("X", "axis_1"),
]
gate_order = GateOrder(descriptors)
```

### Noisy Circuits

Add depolarizing noise after CX gates:

```python
circuit = SyndromeCircuit(
    qubit_system, lattice_points, gate_order,
    num_noisy_cycles=10,  # Total: 2 + 10 = 12 cycles
    p_cx=0.001  # 0.1% depolarizing error per CX
)
```

### Detectors and Observables

Detectors compare ancilla measurements across cycles. Observables define logical operators:

```python
# Detectors are automatically added for stabilizer measurements
# Observables are added if include_observables=True

# The circuit will contain:
# - DETECTOR instructions for stabilizer parity checks
# - OBSERVABLE_INCLUDE instructions for logical operators

stim_circuit = circuit.to_stim_circuit()
# Use with Stim's detector error model generation
dem = stim_circuit.detector_error_model()
```

## Lattice Vector Specification

Lattice vectors define the periodic boundary conditions. They must be:
- Linearly independent
- Integer-valued
- D vectors for a D-dimensional lattice

Examples:
- **Hypercubic**: `[[L, 0, 0], [0, L, 0], [0, 0, L]]` for L×L×L lattice
- **Toric**: `[[L, 0], [0, L]]` for L×L torus
- **Custom**: `[[2, 1], [1, 2]]` for non-orthogonal 2D lattice

## Applications

This framework is designed for:
- Bivariate bicycle quantum code simulation
- Time vortex defect studies
- 4D quantum error correction
- Syndrome extraction circuit generation
- Stabilizer measurement protocols
