"""
Generic threshold plotting module for quantum error correction codes.

This module provides flexible functions to plot logical error rate vs physical error rate
for arbitrary quantum codes, supporting multiple codes on the same axis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any
import csv
import itertools
import sinter
import tesseract_decoder.tesseract as tesseract
import tesseract_decoder.tesseract_sinter_compat as tesseract_sinter

from lattice import Lattice
from qubit_system import QubitSystem
from gate_order import GateOrder, GateDescriptor
from syndrome_circuit import SyndromeCircuit
import random


def calculate_wilson_confidence_interval(errors: int, shots: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Wilson score interval for binomial proportion confidence interval.
    
    Args:
        errors: Number of errors observed
        shots: Total number of shots
        confidence: Confidence level (default: 0.95)
    
    Returns:
        Tuple of (error_rate, error_bar)
    """
    if shots == 0:
        return 0.0, 0.0
    
    p = errors / shots
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99% confidence
    
    # Wilson score interval
    denominator = 1 + z**2 / shots
    centre_adjusted_probability = (p + z**2 / (2*shots)) / denominator
    adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4*shots)) / shots) / denominator
    
    lower_bound = max(0, centre_adjusted_probability - z * adjusted_standard_deviation)
    upper_bound = min(1, centre_adjusted_probability + z * adjusted_standard_deviation)
    error_bar = max(p - lower_bound, upper_bound - p)
    
    return p, error_bar


def plot_code_threshold(
    lattice_vectors: List[List[int]],
    code_name: str,
    gate_order: Optional[GateOrder] = None,
    num_noisy_cycles: int = 2,
    num_noiseless_cycles_init: int = 1,
    num_noiseless_cycles_final: int = 1,
    p_cx_values: List[float] = None,
    p_phenomenological_values: List[float] = None,
    vortex_counts: Optional[List[int]] = None,
    shots_per_point: int = 1000,
    num_workers: int = 4,
    basis: str = 'Z',
    max_errors: int = 100,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    marker: str = 'o'
) -> Dict[str, Any]:
    """
    Plot logical error rate vs physical error rate for a quantum code.
    
    Args:
        lattice_vectors: List of lattice vectors defining the code
        code_name: String label for plot legend
        gate_order: GateOrder object (or None for default based on dimension)
        num_noisy_cycles: Number of noisy cycles
        num_noiseless_cycles_init: Number of initial noiseless cycles (default: 1)
        num_noiseless_cycles_final: Number of final noiseless cycles (default: 1)
        p_cx_values: List of physical error rates to test
        p_phenomenological_values: List of phenomenological error rates to test
        vortex_counts: List of vortex counts (or None)
        shots_per_point: Number of shots per data point
        num_workers: Number of parallel workers
        basis: Measurement basis ('Z' or 'X')
        max_errors: Maximum errors before stopping
        ax: Matplotlib axis to plot on (optional, creates new figure if None)
        color: Color for plot line (optional, auto-assigned if None)
        marker: Marker style (default: 'o')
    
    Returns:
        Dictionary with results data: {'x_values': [...], 'x_label': '...', 'logical_error_rates': [...], 'error_bars': [...]}
    """
    # Determine which noise type is actually being swept
    # If both are provided and non-zero, prefer p_cx for x-axis
    # If only one has non-zero values, use that one
    
    if p_cx_values is None and p_phenomenological_values is None:
        # Default: use p_phenomenological for x-axis
        p_cx_values = [0.0] * 5
        p_phenomenological_values = [0.001, 0.002, 0.003, 0.004, 0.005]
        noise_type = 'p_phenomenological'
        x_values = p_phenomenological_values
    elif p_cx_values is None:
        # Only p_phenomenological is provided
        p_cx_values = [0.0] * len(p_phenomenological_values)
        noise_type = 'p_phenomenological'
        x_values = p_phenomenological_values
    elif p_phenomenological_values is None:
        # Only p_cx is provided
        p_phenomenological_values = [0.0] * len(p_cx_values)
        noise_type = 'p_cx'
        x_values = p_cx_values
    else:
        # Both provided - check which one has non-zero values
        p_cx_nonzero = any(v > 0 for v in p_cx_values)
        p_phenom_nonzero = any(v > 0 for v in p_phenomenological_values)
        
        if p_cx_nonzero and not p_phenom_nonzero:
            noise_type = 'p_cx'
            x_values = p_cx_values
        elif p_phenom_nonzero and not p_cx_nonzero:
            noise_type = 'p_phenomenological'
            x_values = p_phenomenological_values
        elif p_cx_nonzero and p_phenom_nonzero:
            # Both have non-zero, prefer p_cx but warn user
            print("Warning: Both p_cx and p_phenomenological have non-zero values. Using p_cx for x-axis.")
            noise_type = 'p_cx'
            x_values = p_cx_values
        else:
            # Both are all zeros, default to p_cx
            noise_type = 'p_cx'
            x_values = p_cx_values
        
        assert len(p_cx_values) == len(p_phenomenological_values), "p_cx_values and p_phenomenological_values must have the same length"
    
    print(f"\n=== {code_name} Threshold Analysis ===")
    print(f"Lattice vectors: {lattice_vectors}")
    print(f"Noisy cycles: {num_noisy_cycles}")
    print(f"Noiseless cycles: init={num_noiseless_cycles_init}, final={num_noiseless_cycles_final}")
    print(f"X-axis parameter: {noise_type}")
    print(f"p_cx values: {p_cx_values}")
    print(f"p_phenomenological values: {p_phenomenological_values}")
    print(f"Shots per point: {shots_per_point}")
    print(f"Vortex counts: {vortex_counts}")
    
    # Create lattice and qubit system
    lattice = Lattice(lattice_vectors)
    qubit_system = QubitSystem(lattice)
    lattice_points = lattice.get_all_lattice_points()
    
    # Get gate order (use provided or default based on dimension)
    if gate_order is None:
        if lattice.dimension == 2:
            # Use gate order from generate_toric_crumble.py
            gate_order = GateOrder([
                GateDescriptor('X', 'on_site_R'), GateDescriptor('Z', 'axis_0'),
                GateDescriptor('X', 'axis_1'), GateDescriptor('Z', 'on_site_R'),
                GateDescriptor('X', 'on_site_L'), GateDescriptor('Z', 'axis_1'),
                GateDescriptor('X', 'axis_0'), GateDescriptor('Z', 'on_site_L'),
            ])
        else:
            gate_order = GateOrder.get_default_order(lattice.dimension)
    
    # Store results
    logical_error_rates = []
    error_bars = []
    
    # Loop over parameter values
    for i in range(len(p_cx_values)):
        p_cx = p_cx_values[i]
        p_phenomenological = p_phenomenological_values[i]
        print(f"  Point {i+1}/{len(p_cx_values)}: p_cx = {p_cx:.6f}, p_phenomenological = {p_phenomenological:.6f}")
        
        try:
            # Build SyndromeCircuit with all parameters
            circuit = SyndromeCircuit(
                qubit_system, lattice_points, gate_order,
                num_noisy_cycles=num_noisy_cycles,
                num_noiseless_cycles_init=num_noiseless_cycles_init,
                num_noiseless_cycles_final=num_noiseless_cycles_final,
                p_cx=p_cx,
                p_phenomenological=p_phenomenological,
                basis=basis,
                include_observables=True,
                vortex_counts=vortex_counts
            )
            
            # Generate Stim circuit
            stim_circuit = circuit.to_stim_circuit()
            
            # Run Sinter decoding with Tesseract
            dem = stim_circuit.detector_error_model()
            
            def get_tesseract_decoder_for_sinter():
                return {"tesseract": tesseract_sinter.TesseractSinterDecoder()}
            
            results, = sinter.collect(
                num_workers=num_workers,
                tasks=[sinter.Task(circuit=stim_circuit)],
                decoders=["tesseract"],
                max_shots=shots_per_point,
                max_errors=max_errors,
                custom_decoders=get_tesseract_decoder_for_sinter(),
            )
            
            # Calculate logical error rate and confidence interval
            logical_error_rate, error_bar = calculate_wilson_confidence_interval(results.errors, results.shots)
            logical_error_rates.append(logical_error_rate)
            error_bars.append(error_bar)
            
            print(f"    Shots: {results.shots}")
            print(f"    Errors: {results.errors}")
            print(f"    Logical error rate: {logical_error_rate:.6f} ± {error_bar:.6f}")
            
        except Exception as e:
            print(f"    Error at p_cx={p_cx:.6f}: {e}")
            logical_error_rates.append(np.nan)
            error_bars.append(np.nan)
    
    # Plot results
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    
    # Determine x-axis label
    x_label = 'Physical Error Rate (p_cx)' if noise_type == 'p_cx' else 'Physical Error Rate (p_phenomenological)'
    
    # Filter out NaN values for plotting
    valid_indices = ~np.isnan(logical_error_rates)
    valid_x_values = np.array(x_values)[valid_indices]
    valid_rates = np.array(logical_error_rates)[valid_indices]
    valid_errors = np.array(error_bars)[valid_indices]
    
    ax.errorbar(valid_x_values, valid_rates, yerr=valid_errors,
                fmt=f'{marker}-', color=color, linewidth=2, markersize=6,
                capsize=5, capthick=2, elinewidth=2,
                label=code_name)
    
    # Return results data
    return {
        'x_values': x_values,
        'x_label': x_label,
        'noise_type': noise_type,
        'logical_error_rates': logical_error_rates,
        'error_bars': error_bars,
        'code_name': code_name
    }


def example_plot_multiple_codes():
    """
    Demonstrate plotting multiple codes on the same axis.
    
    Plots:
    1. Unrotated toric code d=3 (lattice vectors [[3,0],[0,3]])
    2. Unrotated toric code d=5 (lattice vectors [[5,0],[0,5]])
    3. Rectangular toric code (lattice vectors [[4,0],[0,5]], vortex_counts=[1,0])
    """
    print("=== Example: Multiple Codes on Same Axis ===")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Common parameters
    p_cx_values = None #[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
    p_phenomenological_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    shots_per_point = 1000000
    num_workers = 10
    basis = 'X'
    max_errors = 100
    num_noisy_cycles = 2
    num_noiseless_cycles_init = 2
    num_noiseless_cycles_final = 2
    
    # Plot unrotated toric code d=3
    print("\n--- Plotting Toric Code d=3 ---")
    result1 = plot_code_threshold(
        lattice_vectors=[[3, 0], [0, 3]],
        code_name="Toric d=3",
        num_noisy_cycles=num_noisy_cycles,
        num_noiseless_cycles_init=num_noiseless_cycles_init,
        num_noiseless_cycles_final=num_noiseless_cycles_final,
        p_cx_values=p_cx_values,
        p_phenomenological_values=p_phenomenological_values,
        shots_per_point=shots_per_point,
        num_workers=num_workers,
        basis=basis,
        max_errors=max_errors,
        ax=ax,
        color='blue',
        marker='o'
    )
    
    # Plot unrotated toric code d=5
    print("\n--- Plotting Toric Code d=5 ---")
    plot_code_threshold(
        lattice_vectors=[[5, 0], [0, 5]],
        code_name="Toric d=5",
        num_noisy_cycles=num_noisy_cycles,
        num_noiseless_cycles_init=num_noiseless_cycles_init,
        num_noiseless_cycles_final=num_noiseless_cycles_final,
        p_cx_values=p_cx_values,
        p_phenomenological_values=p_phenomenological_values,
        shots_per_point=shots_per_point,
        num_workers=num_workers,
        basis=basis,
        max_errors=max_errors,
        ax=ax,
        color='red',
        marker='s'
    )
    
    # Plot rectangular toric code with vortices (vortex [1,0])
    print("\n--- Plotting Rectangular Toric Code with Vortices [1,0] ---")
    plot_code_threshold(
        lattice_vectors=[[4, 0], [0, 5]],
        code_name="Rectangular Toric [4,0],[0,5] + vortex [1,0]",
        num_noisy_cycles=num_noisy_cycles,
        num_noiseless_cycles_init=num_noiseless_cycles_init,
        num_noiseless_cycles_final=num_noiseless_cycles_final,
        vortex_counts=[1, 0],
        p_cx_values=p_cx_values,
        p_phenomenological_values=p_phenomenological_values,
        shots_per_point=shots_per_point,
        num_workers=num_workers,
        basis=basis,
        max_errors=max_errors,
        ax=ax,
        color='green',
        marker='^'
    )

    # Plot rectangular toric code with vortices (vortex [-1,0])
    print("\n--- Plotting Rectangular Toric Code with Vortices [-1,0] ---")
    plot_code_threshold(
        lattice_vectors=[[4, 0], [0, 5]],
        code_name="Rectangular Toric [4,0],[0,5] + vortex [-1,0]",
        num_noisy_cycles=num_noisy_cycles,
        num_noiseless_cycles_init=num_noiseless_cycles_init,
        num_noiseless_cycles_final=num_noiseless_cycles_final,
        vortex_counts=[-1, 0],
        p_cx_values=p_cx_values,
        p_phenomenological_values=p_phenomenological_values,
        shots_per_point=shots_per_point,
        num_workers=num_workers,
        basis=basis,
        max_errors=max_errors,
        ax=ax,
        color='orange',
        marker='v'
    )
    
    
    # Configure axis
    ax.set_xlabel(result1['x_label'], fontsize=12)
    ax.set_ylabel('Logical Error Rate', fontsize=12)
    ax.set_title('Quantum Code Threshold Comparison\nLogical Error Rate vs Physical Error Rate', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== Plot Complete ===")


def example_plot_bb_code_with_vortices():
    """
    Demonstrate plotting BB code with different vortex configurations.
    
    Plots:
    1. BB code with no vortices [0,0,0,0]
    2. BB code with vortex in direction 2: [0,0,1,0] and [0,0,-1,0]
    3. BB code with vortex in direction 3: [0,0,0,1] and [0,0,0,-1]
    """
    print("=== Example: BB Code with Vortices ===")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Common parameters - using phenomenological noise instead of p_cx
    p_cx_values = None  # No gate noise
    p_phenomenological_values = [0.02]
    shots_per_point = 3000000  # More shots for better statistics
    num_workers = 10
    basis = 'Z'
    max_errors = 1000
    num_noisy_cycles = 4
    num_noiseless_cycles_init = 3
    num_noiseless_cycles_final = 3
    
    # BB code gate order
    bb_gate_order = GateOrder([
        GateDescriptor("Z", "on_site_L"),
        GateDescriptor("X", "axis_0"), GateDescriptor("Z", "axis_2"),
        GateDescriptor("X", "axis_1"), GateDescriptor("Z", "on_site_R"),
        GateDescriptor("X", "on_site_L"), GateDescriptor("Z", "axis_1"),
        GateDescriptor("X", "axis_3"), GateDescriptor("Z", "axis_3"),
        GateDescriptor("X", "on_site_R"), GateDescriptor("Z", "axis_0"),
        GateDescriptor("X", "axis_2")
    ])

    # BB code gate order same as above but swap L<->R and axis_0<->axis_1 and axis_2<->axis_3
    bb_gate_order_swap = GateOrder([
        GateDescriptor("Z", "on_site_R"),
        GateDescriptor("X", "axis_1"), GateDescriptor("Z", "axis_3"),
        GateDescriptor("X", "axis_0"), GateDescriptor("Z", "on_site_L"),
        GateDescriptor("X", "on_site_R"), GateDescriptor("Z", "axis_0"),
        GateDescriptor("X", "axis_2"), GateDescriptor("Z", "axis_2"),
        GateDescriptor("X", "on_site_L"), GateDescriptor("Z", "axis_1"),
        GateDescriptor("X", "axis_3")
    ])
    
    # Define vortex configurations and their colors/markers
    vortex_configs = [
        ([0, 0, 1, 0], "Vortex +[0,0,1,0]", 'orange', 'D'),
        ([0, 0, -1, 0], "Vortex -[0,0,1,0]", 'darkorange', 'D'),
        ([0, 0, 0, 1], "Vortex +[0,0,0,1]", 'red', 's'),
        ([0, 0, 0, -1], "Vortex -[0,0,0,1]", 'darkred', 's'),
        ([0, 0, 0, 0], "No Vortices", 'blue', 'o'),
    ]
    
    # Plot each vortex configuration
    first_result = None
    for vortex_counts, code_name, color, marker in vortex_configs:
        print(f"\n--- Plotting BB Code: {code_name} ---")
        result = plot_code_threshold(
            lattice_vectors=[[12, 0, 0, 0], [0, 6, 0, 0], [1, 3, 1, 0], [-3, -2, 0, 1]],
            code_name=code_name,
            gate_order=bb_gate_order,
            num_noisy_cycles=num_noisy_cycles,
            num_noiseless_cycles_init=num_noiseless_cycles_init,
            num_noiseless_cycles_final=num_noiseless_cycles_final,
            vortex_counts=vortex_counts,
            p_cx_values=p_cx_values,
            p_phenomenological_values=p_phenomenological_values,
            shots_per_point=shots_per_point,
            num_workers=num_workers,
            basis=basis,
            max_errors=max_errors,
            ax=ax,
            color=color,
            marker=marker
        )
        if first_result is None:
            first_result = result
    
    # Configure axis
    ax.set_xlabel(first_result['x_label'], fontsize=12)
    ax.set_ylabel('Logical Error Rate', fontsize=12)
    ax.set_title('BB Code Threshold Comparison with Vortices\nLogical Error Rate vs Physical Error Rate', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, ncol=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== BB Code Vortex Plot Complete ===")


def test_all_vortex_configs_bb_code(
    p_phenomenological: float = 0.02,
    p_cx: float = 0.0,
    shots_per_point: int = 3000000,
    num_workers: int = 10,
    basis: str = 'Z',
    max_errors: int = 1000,
    num_noisy_cycles: int = 4,
    num_noiseless_cycles_init: int = 3,
    num_noiseless_cycles_final: int = 3,
    output_csv: str = 'bb_code_all_vortex_configs.csv',
    calculate_distance: bool = False
):
    """
    Test all possible vortex configurations for BB code and save results to CSV.
    
    Tests all 3^4 = 81 configurations (each axis can be -1, 0, or 1).
    
    Args:
        p_phenomenological: Phenomenological error rate to test
        p_cx: Depolarizing error probability after CX gates
        shots_per_point: Number of shots per configuration
        num_workers: Number of parallel workers
        basis: Measurement basis ('Z' or 'X')
        max_errors: Maximum errors before stopping
        num_noisy_cycles: Number of noisy cycles
        num_noiseless_cycles_init: Number of initial noiseless cycles
        num_noiseless_cycles_final: Number of final noiseless cycles
        output_csv: Output CSV filename
    """
    print("=== Testing All Vortex Configurations for BB Code ===")
    print(f"Phenomenological noise: p = {p_phenomenological}")
    print(f"CX gate noise: p = {p_cx}")
    print(f"Shots per configuration: {shots_per_point}")
    
    # # Generate all possible vortex configurations (each axis: -1, 0, or 1)
    # vortex_values = [0, -1, 1]
    # all_vortex_configs = list(itertools.product(vortex_values, vortex_values, [0], [0]))
    
    all_vortex_configs = [[0,0,0,0], [1,0,0,0], [-1,0,0,0]]
    
    print(f"Testing {len(all_vortex_configs)} vortex configurations...\n")
    
    # BB code lattice vectors
    # lattice_vectors = [[12, 0, 0, 0], [0, 6, 0, 0], [1, 3, 1, 0], [-3, -2, 0, 1]]
    lattice_vectors = [[0,3,0,0], [3, 0, 0, 0], [-1, -1, 1, 0], [-1, -1, 0, 1]]
    # lattice_vectors = [[0, 6, 0, 0], [1, 2, 0, 0], [0, 5, -1, 0], [0, 2, 0, -1]]
    
    # Create lattice and qubit system
    lattice = Lattice(lattice_vectors)
    qubit_system = QubitSystem(lattice)
    lattice_points = lattice.get_all_lattice_points()
    
    # # BB code gate order
    # bb_gate_order = GateOrder([
    #     GateDescriptor("Z", "on_site_L"),
    #     GateDescriptor("X", "axis_0"), GateDescriptor("Z", "axis_2"),
    #     GateDescriptor("X", "axis_1"), GateDescriptor("Z", "on_site_R"),
    #     GateDescriptor("X", "on_site_L"), GateDescriptor("Z", "axis_1"),
    #     GateDescriptor("X", "axis_3"), GateDescriptor("Z", "axis_3"),
    #     GateDescriptor("X", "on_site_R"), GateDescriptor("Z", "axis_0"),
    #     GateDescriptor("X", "axis_2")
    # ])
    # all Zs then all Xs
    bb_gate_order = GateOrder([
        GateDescriptor("Z", "on_site_L"),
        GateDescriptor("Z", "axis_2"),
        GateDescriptor("Z", "on_site_R"),
        GateDescriptor("Z", "axis_1"),
        GateDescriptor("Z", "axis_3"),
        GateDescriptor("Z", "axis_0"),
        GateDescriptor("X", "axis_0"), 
        GateDescriptor("X", "axis_1"), 
        GateDescriptor("X", "on_site_L"),
        GateDescriptor("X", "axis_3"),
        GateDescriptor("X", "on_site_R"),
        GateDescriptor("X", "axis_2")
    ])

    # all Zs then all Xs in random order
    z_gates = [
        GateDescriptor("Z", "on_site_L"),
        GateDescriptor("Z", "axis_2"),
        GateDescriptor("Z", "on_site_R"),
        GateDescriptor("Z", "axis_1"),
        GateDescriptor("Z", "axis_3"),
        GateDescriptor("Z", "axis_0")
    ]
    x_gates = [
        GateDescriptor("X", "axis_0"), 
        GateDescriptor("X", "axis_1"), 
        GateDescriptor("X", "on_site_L"),
        GateDescriptor("X", "axis_3"),
        GateDescriptor("X", "on_site_R"),
        GateDescriptor("X", "axis_2")
    ]
    random.shuffle(z_gates)
    random.shuffle(x_gates)
    bb_gate_order = GateOrder(z_gates + x_gates)

    # print the gate order
    for gate in bb_gate_order.descriptors:
        print(gate)

    # Store results
    results = []
    
    # CSV fieldnames
    fieldnames = ['vortex_0', 'vortex_1', 'vortex_2', 'vortex_3', 
                 'logical_error_rate', 'error_bar', 'shots', 'errors', 'p_phenomenological', 'p_cx', 'num_logical_qubits', 'distance', 'num_physical_qubits']
    
    # Initialize CSV file with header
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    # Calculate total number of physical qubits (same for all configurations)
    # Each lattice point has 4 qubits: L, R, X_anc, Z_anc
    num_physical_qubits = len(lattice_points) * 4
    
    # Test each vortex configuration
    for i, vortex_counts_tuple in enumerate(all_vortex_configs):
        vortex_counts = list(vortex_counts_tuple)
        print(f"[{i+1}/{len(all_vortex_configs)}] Testing vortex {vortex_counts}...", end=" ")
        
        result = None
        num_logical_qubits = None
        distance = None
        try:
            # Build SyndromeCircuit
            circuit = SyndromeCircuit(
                qubit_system=qubit_system,
                lattice_points=lattice_points,
                gate_order=bb_gate_order,
                num_noisy_cycles=num_noisy_cycles,
                num_noiseless_cycles_init=num_noiseless_cycles_init,
                num_noiseless_cycles_final=num_noiseless_cycles_final,
                p_cx=p_cx,
                p_phenomenological=p_phenomenological,
                basis=basis,
                include_observables=True,
                vortex_counts=vortex_counts
            )
            
            num_logical_qubits = circuit.logical_operators.get_num_logical_qubits()
            print(f"Number of logical qubits: {num_logical_qubits}")
            # Generate Stim circuit
            stim_circuit = circuit.to_stim_circuit()
            if calculate_distance:
                distance = None
                minimal_found_error = stim_circuit.search_for_undetectable_logical_errors(
                    dont_explore_detection_event_sets_with_size_above=6,
                    dont_explore_edges_with_degree_above=9999,
                    dont_explore_edges_increasing_symptom_degree=False,
                    canonicalize_circuit_errors=True)
                distance = len(minimal_found_error)
                print(f"Distance: {distance}")
            else:
                distance = None
            # Run Sinter decoding with Tesseract
            def get_tesseract_decoder_for_sinter():
                return {"tesseract": tesseract_sinter.TesseractSinterDecoder()}
            
            results_sinter, = sinter.collect(
                num_workers=num_workers,
                tasks=[sinter.Task(circuit=stim_circuit)],
                decoders=["tesseract"],
                max_shots=shots_per_point,
                max_errors=max_errors,
                custom_decoders=get_tesseract_decoder_for_sinter(),
            )
            
            # Calculate logical error rate and confidence interval
            logical_error_rate, error_bar = calculate_wilson_confidence_interval(
                results_sinter.errors, results_sinter.shots
            )
            
            # Store result
            result = {
                'vortex_0': vortex_counts[0],
                'vortex_1': vortex_counts[1],
                'vortex_2': vortex_counts[2],
                'vortex_3': vortex_counts[3],
                'logical_error_rate': logical_error_rate,
                'error_bar': error_bar,
                'shots': results_sinter.shots,
                'errors': results_sinter.errors,
                'p_phenomenological': p_phenomenological,
                'p_cx': p_cx,
                'num_logical_qubits': num_logical_qubits,
                'distance': distance,
                'num_physical_qubits': num_physical_qubits
            }
            
            print(f"✓ Error rate: {logical_error_rate:.6f} ± {error_bar:.6f} "
                  f"(shots: {results_sinter.shots}, errors: {results_sinter.errors})")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            # Store result with NaN values
            result = {
                'vortex_0': vortex_counts[0],
                'vortex_1': vortex_counts[1],
                'vortex_2': vortex_counts[2],
                'vortex_3': vortex_counts[3],
                'logical_error_rate': np.nan,
                'error_bar': np.nan,
                'shots': 0,
                'errors': 0,
                'p_phenomenological': p_phenomenological,
                'p_cx': p_cx,
                'num_logical_qubits': num_logical_qubits,
                'distance': distance,
                'num_physical_qubits': num_physical_qubits
            }
        
        # Save result immediately to CSV (append mode)
        if result is not None:
            results.append(result)
            with open(output_csv, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(result)
    
    print(f"\n✓ Saved {len(results)} results to {output_csv}")
    
    # Print error rates and distances
    for result in results:
        print(f"Vortex {result['vortex_0']}, {result['vortex_1']}, {result['vortex_2']}, {result['vortex_3']}: Error rate: {result['logical_error_rate']:.6f}, Distance: {result['distance']}")
        
    print("\n=== All Vortex Configurations Test Complete ===")
    return results


if __name__ == "__main__":
    # example_plot_multiple_codes()
    # Now running BB code example with phenomenological noise
    # example_plot_bb_code_with_vortices()

    test_all_vortex_configs_bb_code(calculate_distance=True, p_cx=0.001, p_phenomenological=0.0)