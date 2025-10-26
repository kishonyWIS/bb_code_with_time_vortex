"""
Generic threshold plotting module for quantum error correction codes.

This module provides flexible functions to plot logical error rate vs physical error rate
for arbitrary quantum codes, supporting multiple codes on the same axis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any
import sinter
import tesseract_decoder.tesseract as tesseract
import tesseract_decoder.tesseract_sinter_compat as tesseract_sinter

from lattice import Lattice
from qubit_system import QubitSystem
from gate_order import GateOrder, GateDescriptor
from syndrome_circuit import SyndromeCircuit


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
        vortex_counts: List of vortex counts (or None)
        shots_per_point: Number of shots per data point
        num_workers: Number of parallel workers
        basis: Measurement basis ('Z' or 'X')
        max_errors: Maximum errors before stopping
        ax: Matplotlib axis to plot on (optional, creates new figure if None)
        color: Color for plot line (optional, auto-assigned if None)
        marker: Marker style (default: 'o')
    
    Returns:
        Dictionary with results data: {'p_cx_values': [...], 'logical_error_rates': [...], 'error_bars': [...]}
    """
    if p_cx_values is None:
        p_cx_values = [0.001, 0.002, 0.003, 0.004, 0.005]
    
    print(f"\n=== {code_name} Threshold Analysis ===")
    print(f"Lattice vectors: {lattice_vectors}")
    print(f"Noisy cycles: {num_noisy_cycles}")
    print(f"Noiseless cycles: init={num_noiseless_cycles_init}, final={num_noiseless_cycles_final}")
    print(f"p_cx values: {p_cx_values}")
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
    
    # Loop over p_cx values
    for i, p_cx in enumerate(p_cx_values):
        print(f"  Point {i+1}/{len(p_cx_values)}: p_cx = {p_cx:.6f}")
        
        try:
            # Build SyndromeCircuit with all parameters
            circuit = SyndromeCircuit(
                qubit_system, lattice_points, gate_order,
                num_noisy_cycles=num_noisy_cycles,
                num_noiseless_cycles_init=num_noiseless_cycles_init,
                num_noiseless_cycles_final=num_noiseless_cycles_final,
                p_cx=p_cx,
                basis=basis,
                include_observables=True,
                vortex_counts=vortex_counts
            )
            
            # Generate Stim circuit
            stim_circuit = circuit.to_stim_circuit()
            
            # Run Sinter decoding with Tesseract
            dem = stim_circuit.detector_error_model()
            tesseract_config = tesseract.TesseractConfig(
                dem=dem,
                pqlimit=200_000,
                det_beam=15,
                beam_climbing=True,
                no_revisit_dets=True,
            )
            
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
            
            print(f"    Logical error rate: {logical_error_rate:.6f} ± {error_bar:.6f}")
            
        except Exception as e:
            print(f"    Error at p_cx={p_cx:.6f}: {e}")
            logical_error_rates.append(np.nan)
            error_bars.append(np.nan)
    
    # Plot results
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))
    
    # Filter out NaN values for plotting
    valid_indices = ~np.isnan(logical_error_rates)
    valid_p_cx = np.array(p_cx_values)[valid_indices]
    valid_rates = np.array(logical_error_rates)[valid_indices]
    valid_errors = np.array(error_bars)[valid_indices]
    
    ax.errorbar(valid_p_cx, valid_rates, yerr=valid_errors,
                fmt=f'{marker}-', color=color, linewidth=2, markersize=6,
                capsize=5, capthick=2, elinewidth=2,
                label=code_name)
    
    # Return results data
    return {
        'p_cx_values': p_cx_values,
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
    p_cx_values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
    shots_per_point = 10000
    num_workers = 10
    basis = 'X'
    max_errors = 100
    num_noisy_cycles = 2
    num_noiseless_cycles_init = 2
    num_noiseless_cycles_final = 2
    
    # Plot unrotated toric code d=3
    print("\n--- Plotting Toric Code d=3 ---")
    plot_code_threshold(
        lattice_vectors=[[3, 0], [0, 3]],
        code_name="Toric d=3",
        num_noisy_cycles=num_noisy_cycles,
        num_noiseless_cycles_init=num_noiseless_cycles_init,
        num_noiseless_cycles_final=num_noiseless_cycles_final,
        p_cx_values=p_cx_values,
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
        shots_per_point=shots_per_point,
        num_workers=num_workers,
        basis=basis,
        max_errors=max_errors,
        ax=ax,
        color='red',
        marker='s'
    )
    
    # Plot rectangular toric code with vortices
    print("\n--- Plotting Rectangular Toric Code with Vortices ---")
    plot_code_threshold(
        lattice_vectors=[[4, 0], [0, 5]],
        code_name="Rectangular Toric [4,0],[0,5] + vortex [1,0]",
        num_noisy_cycles=num_noisy_cycles,
        num_noiseless_cycles_init=num_noiseless_cycles_init,
        num_noiseless_cycles_final=num_noiseless_cycles_final,
        vortex_counts=[1, 0],
        p_cx_values=p_cx_values,
        shots_per_point=shots_per_point,
        num_workers=num_workers,
        basis=basis,
        max_errors=max_errors,
        ax=ax,
        color='green',
        marker='^'
    )
    
    
    # Configure axis
    ax.set_xlabel('Physical Error Rate (p_cx)', fontsize=12)
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
    
    # Common parameters
    p_cx_values = [0.004]#, 0.005, 0.006, 0.007, 0.008]
    shots_per_point = 10000
    num_workers = 10
    basis = 'Z'
    max_errors = 100
    num_noisy_cycles = 12
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
    for vortex_counts, code_name, color, marker in vortex_configs:
        print(f"\n--- Plotting BB Code: {code_name} ---")
        plot_code_threshold(
            lattice_vectors=[[12, 0, 0, 0], [0, 6, 0, 0], [1, 3, 1, 0], [-3, -2, 0, 1]],
            code_name=code_name,
            gate_order=bb_gate_order,
            num_noisy_cycles=num_noisy_cycles,
            num_noiseless_cycles_init=num_noiseless_cycles_init,
            num_noiseless_cycles_final=num_noiseless_cycles_final,
            vortex_counts=vortex_counts,
            p_cx_values=p_cx_values,
            shots_per_point=shots_per_point,
            num_workers=num_workers,
            basis=basis,
            max_errors=max_errors,
            ax=ax,
            color=color,
            marker=marker
        )
    
    # Configure axis
    ax.set_xlabel('Physical Error Rate (p_cx)', fontsize=12)
    ax.set_ylabel('Logical Error Rate', fontsize=12)
    ax.set_title('BB Code Threshold Comparison with Vortices\nLogical Error Rate vs Physical Error Rate', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, ncol=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== BB Code Vortex Plot Complete ===")


if __name__ == "__main__":
    # example_plot_multiple_codes()
    # Uncomment the line below to run the BB code example instead
    example_plot_bb_code_with_vortices()
