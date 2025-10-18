"""
Examples demonstrating the lattice qubit framework for bivariate bicycle quantum code.
"""

import numpy as np
import matplotlib.pyplot as plt
from lattice import Lattice, Point
from qubit_system import QubitSystem
from gate_order import GateOrder, GateDescriptor
from syndrome_circuit import SyndromeCircuit
from typing import List
import sinter
try:
    import tesseract_decoder.tesseract as tesseract
    import tesseract_decoder.tesseract_sinter_compat as tesseract_sinter
    import tesseract_decoder.utils as tesseract_utils
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: tesseract_decoder not available. Will use alternative decoder.")


def toric_code_example(distance: int = 2, noisy_cycles: int = None, rotated: bool = False, basis: str = 'Z', vortex_counts: List[int] = [0, 0]):
    """Example demonstrating final data qubit measurements and logical observables."""
    print("\n=== Final Measurements and Observables Example ===")
    
    if noisy_cycles is None:
        noisy_cycles = distance
        
    lattice_vectors = [[distance, 0], [0, distance]] if not rotated else [[distance, -distance], [distance, distance]]
    lattice = Lattice(lattice_vectors)
    qubit_system = QubitSystem(lattice)
    lattice_points = lattice.get_all_lattice_points()
    gate_order = GateOrder([
        GateDescriptor("Z", "on_site_L"), GateDescriptor("Z", "axis_1"), GateDescriptor("Z", "on_site_R"), GateDescriptor("Z", "axis_0"),
        GateDescriptor("X", "on_site_L"), GateDescriptor("X", "axis_1"), GateDescriptor("X", "on_site_R"), GateDescriptor("X", "axis_0")])
    # gate_order = GateOrder([
    #     GateDescriptor("Z", "on_site_L"), GateDescriptor("Z", "on_site_R"), GateDescriptor("Z", "axis_1"), GateDescriptor("Z", "axis_0"),
    #     GateDescriptor("X", "on_site_L"), GateDescriptor("X", "on_site_R"), GateDescriptor("X", "axis_1"), GateDescriptor("X", "axis_0")])

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
        num_noisy_cycles=noisy_cycles,
        basis=basis,
        include_observables=True,
        include_detectors=True,
        p_cx=0.001,
        vortex_counts=vortex_counts
    )

    # Generate Stim circuit
    stim_circuit = circuit.to_stim_circuit()
    print(stim_circuit)
    # print(stim_circuit.detector_error_model())
    print('number of logical qubits: ', circuit.logical_operators.get_num_logical_qubits())

    print(f'Shortest graphlike error: {len(stim_circuit.shortest_graphlike_error())}')
    minimal_found_error = stim_circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=9999,
        dont_explore_edges_with_degree_above=9999,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True)
    print(f'Minimal found error: {len(minimal_found_error)}')

def bb_code_example(basis: str = 'Z', noisy_cycles: int = 2, vortex_counts: List[int] = [0, 0]):
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
        p_cx=0.001,
        vortex_counts=vortex_counts
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
                                 num_workers: int = 4,
                                 max_errors: int = 100):
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
        max_errors: Maximum number of errors to explore
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
            max_errors=max_errors,
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
            max_errors=max_errors,
        )
    
    # Display results
    print(f"\n=== Decoding Results ===")
    print(f"Shots run: {results.shots}")
    print(f"Errors detected: {results.errors}")
    print(f"Logical error rate: {results.errors / results.shots:.6f}")


def plot_toric_threshold_curve(distances: list = [3, 5, 7],
                              p_cx_values: list = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008], 
                              shots_per_point: int = 1000,
                              num_workers: int = 4,
                              basis: str = 'Z',
                              rotated: bool = False,
                              vortex_counts: List[int] = [0, 0]):
    """
    Plot logical error rate vs p_cx for different toric code distances.
    
    Args:
        distances: List of toric code distances to test
        p_cx_values: List of p_cx values to test
        shots_per_point: Number of shots for each data point
        num_workers: Number of parallel workers for Sinter
        basis: Measurement basis ('Z' or 'X')
        rotated: Whether to use a rotated toric code
    """
    print(f"\n=== Toric Code Threshold Curve Analysis ===")
    print(f"Distances: {distances}")
    print(f"p_cx values: {p_cx_values}")
    print(f"Shots per point: {shots_per_point}")
    print(f"Rotated: {rotated}")
    
    # Store results for each distance
    results_by_distance = {}
    error_bars_by_distance = {}
    
    for distance in distances:
        print(f"\n--- Testing distance {distance} ---")
        logical_error_rates = []
        error_bars = []
        
        for i, p_cx in enumerate(p_cx_values):
            print(f"  Point {i+1}/{len(p_cx_values)}: p_cx = {p_cx:.6f}")
            
            try:
                # Create toric circuit for this distance
                lattice_vectors = [[distance, 0], [0, distance]] if not rotated else [[distance, -distance], [distance, distance]]
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
                    p_cx=p_cx,
                    vortex_counts=vortex_counts
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
                        max_errors=100,
                        custom_decoders=get_tesseract_decoder_for_sinter(),
                    )
                else:
                    # Fallback to built-in decoders
                    results, = sinter.collect(
                        num_workers=num_workers,
                        tasks=[sinter.Task(circuit=stim_circuit)],
                        decoders=["pymatching"],
                        max_shots=shots_per_point,
                        max_errors=100,
                    )
                
                # Calculate logical error rate and error bar (binomial confidence interval)
                logical_error_rate = results.errors / results.shots
                logical_error_rates.append(logical_error_rate)
                
                # Calculate 95% confidence interval using Wilson score interval
                n = results.shots
                k = results.errors
                if n > 0:
                    p = k / n
                    z = 1.96  # 95% confidence interval
                    # Wilson score interval
                    denominator = 1 + z**2 / n
                    centre_adjusted_probability = (p + z**2 / (2*n)) / denominator
                    adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4*n)) / n) / denominator
                    lower_bound = max(0, centre_adjusted_probability - z * adjusted_standard_deviation)
                    upper_bound = min(1, centre_adjusted_probability + z * adjusted_standard_deviation)
                    error_bar = max(logical_error_rate - lower_bound, upper_bound - logical_error_rate)
                else:
                    error_bar = 0
                
                error_bars.append(error_bar)
                
                print(f"    Logical error rate: {logical_error_rate:.6f} ± {error_bar:.6f}")
                
            except Exception as e:
                print(f"    Error at p_cx={p_cx:.6f}: {e}")
                logical_error_rates.append(np.nan)
                error_bars.append(np.nan)
        
        results_by_distance[distance] = logical_error_rates
        error_bars_by_distance[distance] = error_bars
    
    # Create the plot
    plt.figure(figsize=(10, 7))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, distance in enumerate(distances):
        color = colors[i % len(colors)]
        # Filter out NaN values for plotting
        valid_indices = ~np.isnan(results_by_distance[distance])
        valid_p_cx = np.array(p_cx_values)[valid_indices]
        valid_rates = np.array(results_by_distance[distance])[valid_indices]
        valid_errors = np.array(error_bars_by_distance[distance])[valid_indices]
        
        plt.errorbar(valid_p_cx, valid_rates, yerr=valid_errors,
                    fmt='o-', color=color, linewidth=2, markersize=6,
                    capsize=5, capthick=2, elinewidth=2,
                    label=f'd = {distance}')
    
    plt.xlabel('Physical Error Rate (p_cx)', fontsize=12)
    plt.ylabel('Logical Error Rate', fontsize=12)
    plt.title('Toric Code Threshold Curve\nLogical Error Rate vs Physical Error Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n=== Summary ===")
    for distance in distances:
        print(f"Distance {distance}:")
        for i, p_cx in enumerate(p_cx_values):
            if not np.isnan(results_by_distance[distance][i]):
                error_bar = error_bars_by_distance[distance][i]
                print(f"  p_cx = {p_cx:.6f}: Logical error rate = {results_by_distance[distance][i]:.6f} ± {error_bar:.6f}")


def plot_bb_threshold_curve(p_cx_values: list = [0.001, 0.002, 0.003, 0.004], 
                           shots_per_point: int = 1000,
                           num_workers: int = 4,
                           basis: str = 'Z',
                           noisy_cycles: int = 2,
                           max_errors: int = 100,
                           vortex_counts: List[int] = [0, 0, 0, 0]):
    """
    Plot logical error rate vs p_cx for BB.
    
    Args:
        p_cx_values: List of p_cx values to test
        shots_per_point: Number of shots for each data point
        num_workers: Number of parallel workers for Sinter
        basis: Measurement basis ('Z' or 'X')
        noisy_cycles: Number of noisy cycles for BB code
    """
    print(f"\n=== BB Code Threshold Curve Analysis ===")
    print(f"p_cx values: {p_cx_values}")
    print(f"Shots per point: {shots_per_point}")
    print(f"Noisy cycles: {noisy_cycles}")
    
    # Store results
    logical_error_rates = []
    error_bars = []
    
    print(f"\n--- Testing BB Code ---")
    
    for i, p_cx in enumerate(p_cx_values):
        print(f"  Point {i+1}/{len(p_cx_values)}: p_cx = {p_cx:.6f}")
        
        try:
            # Create BB circuit
            lattice_vectors = [[12,0,0,0], [0, 6, 0, 0], [1, 3, 1, 0], [-3, -2, 0, 1]]
            lattice = Lattice(lattice_vectors)
            qubit_system = QubitSystem(lattice)
            lattice_points = lattice.get_all_lattice_points()
            gate_order = GateOrder.get_default_order(lattice.dimension)
            
            # Create circuit with detectors and observables
            circuit = SyndromeCircuit(
                qubit_system, lattice_points, gate_order,
                num_noisy_cycles=noisy_cycles,
                basis=basis,
                include_observables=True,
                include_detectors=True,
                p_cx=p_cx,
                vortex_counts=vortex_counts
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
                    max_errors=max_errors,
                )
            else:
                # Fallback to built-in decoders
                results, = sinter.collect(
                    num_workers=num_workers,
                    tasks=[sinter.Task(circuit=stim_circuit)],
                    decoders=["pymatching"],
                    max_shots=shots_per_point,
                    max_errors=max_errors,
                )
            
            # Calculate logical error rate and error bar (binomial confidence interval)
            logical_error_rate = results.errors / results.shots
            logical_error_rates.append(logical_error_rate)
            
            # Calculate 95% confidence interval using Wilson score interval
            n = results.shots
            k = results.errors
            if n > 0:
                p = k / n
                z = 1.96  # 95% confidence interval
                # Wilson score interval
                denominator = 1 + z**2 / n
                centre_adjusted_probability = (p + z**2 / (2*n)) / denominator
                adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4*n)) / n) / denominator
                lower_bound = max(0, centre_adjusted_probability - z * adjusted_standard_deviation)
                upper_bound = min(1, centre_adjusted_probability + z * adjusted_standard_deviation)
                error_bar = max(logical_error_rate - lower_bound, upper_bound - logical_error_rate)
            else:
                error_bar = 0
            
            error_bars.append(error_bar)
            
            print(f"    Logical error rate: {logical_error_rate:.6f} ± {error_bar:.6f}")
            
        except Exception as e:
            print(f"    Error at p_cx={p_cx:.6f}: {e}")
            logical_error_rates.append(np.nan)
            error_bars.append(np.nan)
    
    # Create the plot
    plt.figure(figsize=(10, 7))
    
    # Filter out NaN values for plotting
    valid_indices = ~np.isnan(logical_error_rates)
    valid_p_cx = np.array(p_cx_values)[valid_indices]
    valid_rates = np.array(logical_error_rates)[valid_indices]
    valid_errors = np.array(error_bars)[valid_indices]
    
    plt.errorbar(valid_p_cx, valid_rates, yerr=valid_errors,
                fmt='o-', color='purple', linewidth=2, markersize=6,
                capsize=5, capthick=2, elinewidth=2,
                label='BB Code')
    
    plt.xlabel('Physical Error Rate (p_cx)', fontsize=12)
    plt.ylabel('Logical Error Rate', fontsize=12)
    plt.title('BB Code Threshold Curve\nLogical Error Rate vs Physical Error Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"BB Code:")
    for i, p_cx in enumerate(p_cx_values):
        if not np.isnan(logical_error_rates[i]):
            error_bar = error_bars[i]
            print(f"  p_cx = {p_cx:.6f}: Logical error rate = {logical_error_rates[i]:.6f} ± {error_bar:.6f}")


if __name__ == "__main__":
    toric_code_example(distance=4, noisy_cycles=1, rotated=False, basis='Z', vortex_counts=[0, -1])
    # bb_code_example(basis='Z', noisy_cycles=1, vortex_counts=[0, 0, 0, 0])
    
    # Tesseract decoder integration examples:
    # decode_with_tesseract_example(circuit_type='toric', distance=2, shots=1000, p_cx=0.001)
    # decode_with_tesseract_example(circuit_type='bb', shots=1000, p_cx=0.001)
    
    # # Plot threshold curve for toric code
    # plot_toric_threshold_curve(distances=[3, 5], p_cx_values=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008], 
    #                           shots_per_point=10000, num_workers=10, vortex_counts=[0, 0])
    
    # # Plot threshold curve for BB code
    # plot_bb_threshold_curve(p_cx_values=[0.005,0.006,0.007,0.008], 
    #                        shots_per_point=10000, num_workers=10, vortex_counts=[0, 0, 0, 0])