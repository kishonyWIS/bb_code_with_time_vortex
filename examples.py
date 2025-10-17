"""
Examples demonstrating the lattice qubit framework for bivariate bicycle quantum code.
"""

import numpy as np
from lattice import Lattice, Point
from qubit_system import QubitSystem
from gate_order import GateOrder, GateDescriptor
from syndrome_circuit import SyndromeCircuit


def toric_code_example(distance: int = 2, rotated: bool = False, basis: str = 'Z'):
    """Example demonstrating final data qubit measurements and logical observables."""
    print("\n=== Final Measurements and Observables Example ===")
    
    lattice_vectors = [[distance, 0], [0, distance]] if not rotated else [[distance, -distance], [distance, distance]]
    lattice = Lattice(lattice_vectors)
    qubit_system = QubitSystem(lattice)
    lattice_points = lattice.get_all_lattice_points()
    # gate_order = GateOrder([
    #     GateDescriptor("Z", "on_site_L"), GateDescriptor("Z", "axis_0"), GateDescriptor("Z", "on_site_R"), GateDescriptor("Z", "axis_1"),
    #     GateDescriptor("X", "on_site_L"), GateDescriptor("X", "axis_0"), GateDescriptor("X", "on_site_R"), GateDescriptor("X", "axis_1")])
    gate_order = GateOrder([
        GateDescriptor("Z", "on_site_L"), GateDescriptor("Z", "axis_1"), GateDescriptor("Z", "on_site_R"), GateDescriptor("Z", "axis_0"),
        GateDescriptor("X", "on_site_L"), GateDescriptor("X", "axis_1"), GateDescriptor("X", "on_site_R"), GateDescriptor("X", "axis_0")])

    print('=== Qubit Index to Lattice Point Mapping ===')
    for i, point in enumerate(lattice_points):
        all_qubits = qubit_system.get_all_qubits_at_point(point)
        print(f'Point {i} ({point}):')
        for label, idx in all_qubits.items():
            print(f'  Qubit {idx}: {label}')

    print(f'\\n=== Circuit with Detectors ===')

    # Create circuit WITH detectors
    circuit = SyndromeCircuit(
        qubit_system, lattice_points, gate_order,
        num_noisy_cycles=distance,
        basis=basis,
        include_observables=True,
        include_detectors=True,
        p_cx=0.001
    )

    # Generate Stim circuit
    stim_circuit = circuit.to_stim_circuit()
    print(stim_circuit)
    print(stim_circuit.detector_error_model())
    print('number of logical qubits: ', circuit.logical_operators.get_num_logical_qubits())

    print(f'Shortest graphlike error: {len(stim_circuit.shortest_graphlike_error())}')
    minimal_found_error = stim_circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=9999,
        dont_explore_edges_with_degree_above=9999,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True)
    print(f'Minimal found error: {len(minimal_found_error)}')

def bb_code_example(basis: str = 'Z', noisy_cycles: int = 2):
    """Example demonstrating the BB code."""
    print("\n=== BB Code Example ===")
    lattice_vectors = [[12,0,0,0], [0, 6, 0, 0], [1, 3, 1, 0], [-3, -2, 0, 1]]
    # lattice_vectors = [[12,0,0,0], [0, 6, 0, 0], [1, 3, -1, 0], [-3, -1, 0, -1]] # from shoham
    print('volume of the lattice: ', np.linalg.det(lattice_vectors))
    lattice = Lattice(lattice_vectors)
    qubit_system = QubitSystem(lattice)
    lattice_points = lattice.get_all_lattice_points()
    gate_order = GateOrder.get_default_order(lattice.dimension)

    print('\n=== Qubit Index to Lattice Point Mapping ===')
    for i, point in enumerate(lattice_points):
        all_qubits = qubit_system.get_all_qubits_at_point(point)
        print(f'Point {i} ({point}):')
        for label, idx in all_qubits.items():
            print(f'  Qubit {idx}: {label}')

    print(f'\n=== Circuit with Detectors ===')

    # Create circuit WITH detectors
    circuit = SyndromeCircuit(
        qubit_system, lattice_points, gate_order,
        num_noisy_cycles=noisy_cycles,
        basis=basis,
        include_observables=True,
        include_detectors=True,
        p_cx=0.001
    )

    # Generate Stim circuit
    stim_circuit = circuit.to_stim_circuit()
    print(stim_circuit)
    print(stim_circuit.detector_error_model())
    print('number of logical qubits: ', circuit.logical_operators.get_num_logical_qubits())
    minimal_found_error = stim_circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=9999,
        dont_explore_edges_with_degree_above=4,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True)
    print(f'Minimal found error: {len(minimal_found_error)}')

if __name__ == "__main__":
    # toric_code_example(distance=2, rotated=True, basis='Z')
    bb_code_example(basis='Z', noisy_cycles=1)