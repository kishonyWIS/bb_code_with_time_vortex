"""
Examples demonstrating the lattice qubit framework for bivariate bicycle quantum code.
"""

import numpy as np
import matplotlib.pyplot as plt
from lattice import Lattice, Point
from qubit_system import QubitSystem
from gate_order import GateOrder, GateDescriptor
from syndrome_circuit import SyndromeCircuit
import sinter
try:
    import tesseract_decoder.tesseract as tesseract
    import tesseract_decoder.tesseract_sinter_compat as tesseract_sinter
    import tesseract_decoder.utils as tesseract_utils
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: tesseract_decoder not available. Will use alternative decoder.")


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

def decode_with_tesseract_example(circuit_type: str = 'toric', distance: int = 2, 
                                 shots: int = 10000, p_cx: float = 0.001, 
                                 basis: str = 'Z', noisy_cycles: int = None,
                                 num_workers: int = 4):
    """
    Example demonstrating Tesseract decoder integration for computing logical error rates.
    
    Args:
        circuit_type: Type of circuit ('toric' or 'bb')
        distance: Distance parameter for toric code
        shots: Number of shots to run
        p_cx: Depolarizing error probability after CX gates
        basis: Measurement basis ('Z' or 'X')
        noisy_cycles: Number of noisy cycles (None for auto)
        num_workers: Number of parallel workers for Sinter
    """
    print(f"\n=== Tesseract Decoder Integration Example ===")
    print(f"Circuit type: {circuit_type}")
    print(f"Distance: {distance}")
    print(f"Shots: {shots}")
    print(f"Error rate (p_cx): {p_cx}")
    print(f"Basis: {basis}")
    
    # Create circuit based on type
    if circuit_type == 'toric':
        lattice_vectors = [[distance, -distance], [distance, distance]]
        lattice = Lattice(lattice_vectors)
        qubit_system = QubitSystem(lattice)
        lattice_points = lattice.get_all_lattice_points()
        gate_order = GateOrder([
            GateDescriptor("Z", "on_site_L"), GateDescriptor("Z", "axis_1"), 
            GateDescriptor("Z", "on_site_R"), GateDescriptor("Z", "axis_0"),
            GateDescriptor("X", "on_site_L"), GateDescriptor("X", "axis_1"), 
            GateDescriptor("X", "on_site_R"), GateDescriptor("X", "axis_0")])
        if noisy_cycles is None:
            noisy_cycles = distance
    elif circuit_type == 'bb':
        lattice_vectors = [[12,0,0,0], [0, 6, 0, 0], [1, 3, 1, 0], [-3, -2, 0, 1]]
        lattice = Lattice(lattice_vectors)
        qubit_system = QubitSystem(lattice)
        lattice_points = lattice.get_all_lattice_points()
        gate_order = GateOrder.get_default_order(lattice.dimension)
        if noisy_cycles is None:
            noisy_cycles = 2
    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")
    
    print(f"Noisy cycles: {noisy_cycles}")
    
    # Create circuit with detectors and observables
    circuit = SyndromeCircuit(
        qubit_system, lattice_points, gate_order,
        num_noisy_cycles=noisy_cycles,
        basis=basis,
        include_observables=True,
        include_detectors=True,
        p_cx=p_cx
    )
    
    # Generate Stim circuit
    stim_circuit = circuit.to_stim_circuit()
    print(f"Number of logical qubits: {circuit.logical_operators.get_num_logical_qubits()}")
    
    # Get detector error model
    dem = stim_circuit.detector_error_model()
    print(f"Detector error model has {len(dem)} detectors")
    
    print(f"\nRunning {shots} shots with {num_workers} workers...")
    
    # Run decoding with Sinter
    if TESSERACT_AVAILABLE:
        # Configure Tesseract decoder with recommended short-beam setup
        tesseract_config = tesseract.TesseractConfig(
            dem=dem,
            pqlimit=200_000,
            det_beam=15,
            beam_climbing=True,
            no_revisit_dets=True,
        )
        
        # Create Sinter-compatible Tesseract decoder
        def get_tesseract_decoder_for_sinter():
            return {"tesseract": tesseract_sinter.TesseractSinterDecoder()}
        
        print("Using Tesseract decoder...")
        results, = sinter.collect(
            num_workers=num_workers,
            tasks=[sinter.Task(circuit=stim_circuit)],
            decoders=["tesseract"],
            max_shots=shots,
            custom_decoders=get_tesseract_decoder_for_sinter(),
        )
    else:
        # Fallback to built-in decoders
        print("Using built-in decoders (pymatching)...")
        results, = sinter.collect(
            num_workers=num_workers,
            tasks=[sinter.Task(circuit=stim_circuit)],
            decoders=["pymatching"],
            max_shots=shots,
        )
    
    # Display results
    print(f"\n=== Decoding Results ===")
    print(f"Shots run: {results.shots}")
    print(f"Errors detected: {results.errors}")
    print(f"Logical error rate: {results.errors / results.shots:.6f}")


def plot_toric_threshold_curve(distances: list = [3, 5, 7], 
                              p_cx_range: tuple = (0.001, 0.02), 
                              num_points: int = 8,
                              shots_per_point: int = 1000,
                              num_workers: int = 4,
                              basis: str = 'Z'):
    """
    Plot logical error rate vs p_cx for different toric code distances.
    
    Args:
        distances: List of toric code distances to test
        p_cx_range: Tuple of (min_p_cx, max_p_cx) for the error rate range
        num_points: Number of p_cx values to test
        shots_per_point: Number of shots for each data point
        num_workers: Number of parallel workers for Sinter
        basis: Measurement basis ('Z' or 'X')
    """
    print(f"\n=== Toric Code Threshold Curve Analysis ===")
    print(f"Distances: {distances}")
    print(f"p_cx range: {p_cx_range}")
    print(f"Points per curve: {num_points}")
    print(f"Shots per point: {shots_per_point}")
    
    # Generate p_cx values
    p_cx_values = np.logspace(np.log10(p_cx_range[0]), np.log10(p_cx_range[1]), num_points)
    
    # Store results for each distance
    results_by_distance = {}
    
    for distance in distances:
        print(f"\n--- Testing distance {distance} ---")
        logical_error_rates = []
        
        for i, p_cx in enumerate(p_cx_values):
            print(f"  Point {i+1}/{num_points}: p_cx = {p_cx:.6f}")
            
            try:
                # Create toric circuit for this distance
                lattice_vectors = [[distance, 0], [0, distance]]
                lattice = Lattice(lattice_vectors)
                qubit_system = QubitSystem(lattice)
                lattice_points = lattice.get_all_lattice_points()
                gate_order = GateOrder([
                    GateDescriptor("Z", "on_site_L"), GateDescriptor("Z", "axis_1"),
                    GateDescriptor("Z", "on_site_R"), GateDescriptor("Z", "axis_0"),
                    GateDescriptor("X", "on_site_L"), GateDescriptor("X", "axis_1"),
                    GateDescriptor("X", "on_site_R"), GateDescriptor("X", "axis_0")])
                
                # Create circuit with detectors and observables
                circuit = SyndromeCircuit(
                    qubit_system, lattice_points, gate_order,
                    num_noisy_cycles=distance,
                    basis=basis,
                    include_observables=True,
                    include_detectors=True,
                    p_cx=p_cx
                )
                
                # Generate Stim circuit
                stim_circuit = circuit.to_stim_circuit()
                
                # Run decoding
                if TESSERACT_AVAILABLE:
                    # Create Sinter-compatible Tesseract decoder
                    def get_tesseract_decoder_for_sinter():
                        return {"tesseract": tesseract_sinter.TesseractSinterDecoder()}
                    
                    results, = sinter.collect(
                        num_workers=num_workers,
                        tasks=[sinter.Task(circuit=stim_circuit)],
                        decoders=["tesseract"],
                        max_shots=shots_per_point,
                        custom_decoders=get_tesseract_decoder_for_sinter(),
                    )
                else:
                    # Fallback to built-in decoders
                    results, = sinter.collect(
                        num_workers=num_workers,
                        tasks=[sinter.Task(circuit=stim_circuit)],
                        decoders=["pymatching"],
                        max_shots=shots_per_point,
                    )
                
                # Calculate logical error rate
                logical_error_rate = results.errors / results.shots
                logical_error_rates.append(logical_error_rate)
                
                print(f"    Logical error rate: {logical_error_rate:.6f}")
                
            except Exception as e:
                print(f"    Error at p_cx={p_cx:.6f}: {e}")
                logical_error_rates.append(np.nan)
        
        results_by_distance[distance] = logical_error_rates
    
    # Create the plot
    plt.figure(figsize=(10, 7))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, distance in enumerate(distances):
        color = colors[i % len(colors)]
        plt.semilogy(p_cx_values, results_by_distance[distance], 
                    'o-', color=color, linewidth=2, markersize=6,
                    label=f'd = {distance}')
    
    plt.xlabel('Physical Error Rate (p_cx)', fontsize=12)
    plt.ylabel('Logical Error Rate', fontsize=12)
    plt.title('Toric Code Threshold Curve\nLogical Error Rate vs Physical Error Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add threshold line at p_cx = 0.01 (approximate toric code threshold)
    plt.axvline(x=0.01, color='black', linestyle='--', alpha=0.7, label='Threshold ~0.01')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n=== Summary ===")
    for distance in distances:
        print(f"Distance {distance}:")
        for i, p_cx in enumerate(p_cx_values):
            if not np.isnan(results_by_distance[distance][i]):
                print(f"  p_cx = {p_cx:.6f}: Logical error rate = {results_by_distance[distance][i]:.6f}")


if __name__ == "__main__":
    # toric_code_example(distance=2, rotated=True, basis='Z')
    # bb_code_example(basis='Z', noisy_cycles=1)
    
    # Tesseract decoder integration examples:
    # decode_with_tesseract_example(circuit_type='toric', distance=2, shots=1000, p_cx=0.001)
    # decode_with_tesseract_example(circuit_type='bb', shots=1000, p_cx=0.001)
    
    # Plot threshold curve for toric code
    plot_toric_threshold_curve(distances=[3, 5], p_cx_range=(0.001, 0.015), 
                              num_points=6, shots_per_point=10000)