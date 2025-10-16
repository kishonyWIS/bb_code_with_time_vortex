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
    gate_order = GateOrder([
        GateDescriptor("Z", "on_site_L"), GateDescriptor("Z", "axis_0"), GateDescriptor("Z", "on_site_R"), GateDescriptor("Z", "axis_1"),
        GateDescriptor("X", "on_site_L"), GateDescriptor("X", "axis_0"), GateDescriptor("X", "on_site_R"), GateDescriptor("X", "axis_1")])

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
        p_cx=0.01
    )

    # Generate Stim circuit
    stim_circuit = circuit.to_stim_circuit()
    print(stim_circuit)
    print(stim_circuit.detector_error_model())
    print(f'Shortest graphlike error: {len(stim_circuit.shortest_graphlike_error())}')
    minimal_found_error = stim_circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=9999,
        dont_explore_edges_with_degree_above=9999,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=False)
    print(f'Minimal found error: {len(minimal_found_error)}')

if __name__ == "__main__":
    # example_2d_toric_lattice()
    # example_4d_hypercubic_lattice()
    # example_custom_lattice()
    # example_stim_circuit_generation()
    # example_toric_code_syndrome_extraction()
    # example_custom_gate_order()
    toric_code_example(2, rotated=True, basis='Z')
