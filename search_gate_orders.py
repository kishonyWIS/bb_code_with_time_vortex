"""
Search for optimal gate orders with random permutations on a fixed lattice.

Samples random permutations of the standard BB gate order to find configurations
with best distance and logical error rates. Tests each gate order with multiple
vortex configurations, tracking best results for both arbitrary and zero vortex cases.
"""

import numpy as np
import csv
import random
import itertools
import ast
import sys
import os
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
import sinter
import tesseract_decoder.tesseract_sinter_compat as tesseract_sinter

from lattice import Lattice
from qubit_system import QubitSystem
from gate_order import GateOrder, GateDescriptor
from syndrome_circuit import SyndromeCircuit
from plot_threshold import calculate_wilson_confidence_interval


def get_standard_bb_descriptors() -> List[GateDescriptor]:
    """
    Get the standard BB code gate descriptors (not yet ordered).
    
    Returns:
        List of 12 GateDescriptor objects (6 Z, 6 X)
    """
    return [
        GateDescriptor("Z", "on_site_L"),
        GateDescriptor("Z", "on_site_R"),
        GateDescriptor("Z", "axis_0"),
        GateDescriptor("Z", "axis_1"),
        GateDescriptor("Z", "axis_2"),
        GateDescriptor("Z", "axis_3"),
        GateDescriptor("X", "on_site_L"),
        GateDescriptor("X", "on_site_R"),
        GateDescriptor("X", "axis_0"),
        GateDescriptor("X", "axis_1"),
        GateDescriptor("X", "axis_2"),
        GateDescriptor("X", "axis_3"),
    ]


def get_standard_bb_gate_order() -> GateOrder:
    """
    Get the standard BB gate order (interleaved Z and X).
    
    Returns:
        GateOrder with standard BB ordering
    """
    return GateOrder([
        GateDescriptor("Z", "on_site_L"),
        GateDescriptor("X", "axis_0"), GateDescriptor("Z", "axis_2"),
        GateDescriptor("X", "axis_1"), GateDescriptor("Z", "on_site_R"),
        GateDescriptor("X", "on_site_L"), GateDescriptor("Z", "axis_1"),
        GateDescriptor("X", "axis_3"), GateDescriptor("Z", "axis_3"),
        GateDescriptor("X", "on_site_R"), GateDescriptor("Z", "axis_0"),
        GateDescriptor("X", "axis_2")
    ])


def generate_random_gate_order(seed: Optional[int] = None) -> Tuple[GateOrder, List[int]]:
    """
    Generate a random permutation of the gate descriptors.
    
    Args:
        seed: Optional random seed for reproducibility
        
    Returns:
        Tuple of (GateOrder, permutation_indices)
    """
    if seed is not None:
        random.seed(seed)
    
    descriptors = get_standard_bb_descriptors()
    indices = list(range(len(descriptors)))
    random.shuffle(indices)
    
    permuted_descriptors = [descriptors[i] for i in indices]
    return GateOrder(permuted_descriptors), indices


def permutation_to_string(permutation: List[int]) -> str:
    """Convert permutation indices to a string representation."""
    return ','.join(map(str, permutation))


def string_to_permutation(s: str) -> List[int]:
    """Convert string representation back to permutation indices."""
    return list(map(int, s.split(',')))


def gate_order_to_string(gate_order: GateOrder) -> str:
    """Convert gate order to a human-readable string."""
    parts = []
    for desc in gate_order.descriptors:
        parts.append(f"{desc.ancilla_type}:{desc.connection_type}")
    return ' -> '.join(parts)


def search_gate_orders(
    lattice_vectors: List[List[int]],
    p_phenomenological: float = 0.02,
    p_cx: float = 0.0,
    shots_per_point: int = 100000,
    num_samples: int = 1000,
    num_workers: int = 10,
    max_errors: int = 100,
    num_noisy_cycles: int = 4,
    num_noiseless_cycles_init: int = 3,
    num_noiseless_cycles_final: int = 3,
    basis: str = 'Z',
    output_csv: str = 'gate_order_search_results.csv',
    summary_csv: str = 'best_gate_orders_summary.csv',
    checkpoint_interval: int = 1,
    random_seed: Optional[int] = None,
    include_standard: bool = True
):
    """
    Search over random gate order permutations for a fixed lattice.
    Tests each gate order with multiple vortex configurations.
    
    Args:
        lattice_vectors: Fixed lattice vectors [[a0,a1,a2,a3], ...]
        p_phenomenological: Phenomenological error rate
        p_cx: CNOT gate error rate
        shots_per_point: Number of shots per configuration
        num_samples: Number of random permutations to sample
        num_workers: Number of parallel workers for sinter
        max_errors: Maximum errors before stopping
        num_noisy_cycles: Number of noisy cycles
        num_noiseless_cycles_init: Number of initial noiseless cycles
        num_noiseless_cycles_final: Number of final noiseless cycles
        basis: Measurement basis ('Z' or 'X')
        output_csv: Output CSV file for detailed results
        summary_csv: Output CSV file for best configurations
        checkpoint_interval: Save checkpoint every N samples
        random_seed: Random seed for reproducibility
        include_standard: Include standard BB gate order in search
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Generate all vortex configurations (same as search_codes_with_vortices.py)
    vortex_values = [0, -1, -2, 1, 2]
    all_vortex_configs = list(itertools.product(vortex_values, vortex_values, [0], [0]))
    
    print("=== Gate Order Search with Random Permutations ===")
    print(f"Lattice vectors: {lattice_vectors}")
    print(f"Phenomenological noise: p = {p_phenomenological}")
    print(f"CNOT error rate: p_cx = {p_cx}")
    print(f"Shots per configuration: {shots_per_point}")
    print(f"Number of random gate order samples: {num_samples}")
    print(f"Testing {len(all_vortex_configs)} vortex configurations per gate order")
    print(f"Basis: {basis}")
    print()
    
    # Create lattice and qubit system
    lattice = Lattice(lattice_vectors)
    qubit_system = QubitSystem(lattice)
    lattice_points = lattice.get_all_lattice_points()
    
    # Calculate n (physical qubits)
    det = abs(np.linalg.det(np.array(lattice_vectors, dtype=int)))
    n = int(4 * det)
    print(f"Physical qubits (n): {n}")
    
    # Best configurations tracking (like search_codes_with_vortices.py)
    best_by_distance_arbitrary: Optional[Dict[str, Any]] = None
    best_by_distance_zero_vortex: Optional[Dict[str, Any]] = None
    best_by_error_rate_arbitrary: Optional[Dict[str, Any]] = None
    best_by_error_rate_zero_vortex: Optional[Dict[str, Any]] = None
    
    # Detailed results
    all_results = []
    
    # CSV fieldnames
    result_fieldnames = [
        'sample_idx', 'permutation', 'gate_order_str',
        'vortex_0', 'vortex_1', 'vortex_2', 'vortex_3',
        'n', 'k', 'distance', 'logical_error_rate', 'error_bar',
        'shots', 'errors', 'p_phenomenological', 'p_cx',
        'is_standard_order'
    ]
    
    # Initialize output CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result_fieldnames)
        writer.writeheader()
    
    # Statistics
    total_tested = 0
    successful = 0
    failed = 0
    tested_permutations = set()
    
    # Generate gate orders to test
    gate_orders_to_test = []
    
    # Include standard BB order if requested
    if include_standard:
        standard_order = get_standard_bb_gate_order()
        # Get the permutation that corresponds to standard order
        standard_descriptors = get_standard_bb_descriptors()
        standard_perm = []
        for desc in standard_order.descriptors:
            for i, std_desc in enumerate(standard_descriptors):
                if desc.ancilla_type == std_desc.ancilla_type and \
                   desc.connection_type == std_desc.connection_type:
                    standard_perm.append(i)
                    break
        gate_orders_to_test.append((standard_order, standard_perm, True))
        tested_permutations.add(tuple(standard_perm))
    
    # Generate random permutations
    while len(gate_orders_to_test) < num_samples + (1 if include_standard else 0):
        gate_order, perm = generate_random_gate_order()
        perm_tuple = tuple(perm)
        if perm_tuple not in tested_permutations:
            tested_permutations.add(perm_tuple)
            gate_orders_to_test.append((gate_order, perm, False))
    
    print(f"Testing {len(gate_orders_to_test)} gate order configurations")
    print(f"Total configurations (gate orders x vortices): {len(gate_orders_to_test) * len(all_vortex_configs)}")
    print()
    
    # Search loop over gate orders
    for sample_idx, (gate_order, permutation, is_standard) in enumerate(
        tqdm(gate_orders_to_test, desc="Gate Orders")
    ):
        # Test each vortex configuration for this gate order
        for vortex_counts_tuple in all_vortex_configs:
            vortex_counts = list(vortex_counts_tuple)
            total_tested += 1
            
            try:
                # Build SyndromeCircuit
                circuit = SyndromeCircuit(
                    qubit_system=qubit_system,
                    lattice_points=lattice_points,
                    gate_order=gate_order,
                    num_noisy_cycles=num_noisy_cycles,
                    num_noiseless_cycles_init=num_noiseless_cycles_init,
                    num_noiseless_cycles_final=num_noiseless_cycles_final,
                    p_cx=p_cx,
                    p_phenomenological=p_phenomenological,
                    basis=basis,
                    include_observables=True,
                    vortex_counts=vortex_counts
                )
                
                # Calculate k (logical qubits)
                k = circuit.logical_operators.get_num_logical_qubits()
                
                # Generate Stim circuit
                stim_circuit = circuit.to_stim_circuit()
                
                # Calculate distance
                distance = None
                try:
                    minimal_found_error = stim_circuit.search_for_undetectable_logical_errors(
                        dont_explore_detection_event_sets_with_size_above=6,
                        dont_explore_edges_with_degree_above=9999,
                        dont_explore_edges_increasing_symptom_degree=False,
                        canonicalize_circuit_errors=True
                    )
                    distance = len(minimal_found_error)
                except Exception as e:
                    print(f"Warning: Distance calculation failed for sample {sample_idx}, "
                          f"vortex {vortex_counts}: {e}")
                    distance = None
                
                # Calculate logical error rate
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
                    'sample_idx': sample_idx,
                    'permutation': permutation_to_string(permutation),
                    'gate_order_str': gate_order_to_string(gate_order),
                    'vortex_0': vortex_counts[0],
                    'vortex_1': vortex_counts[1],
                    'vortex_2': vortex_counts[2],
                    'vortex_3': vortex_counts[3],
                    'n': n,
                    'k': k,
                    'distance': distance,
                    'logical_error_rate': logical_error_rate,
                    'error_bar': error_bar,
                    'shots': results_sinter.shots,
                    'errors': results_sinter.errors,
                    'p_phenomenological': p_phenomenological,
                    'p_cx': p_cx,
                    'is_standard_order': is_standard
                }
                
                all_results.append(result)
                
                # Determine if this is a zero vortex configuration
                is_zero_vortex = (vortex_counts == [0, 0, 0, 0])
                
                # Update best by distance
                if distance is not None:
                    if best_by_distance_arbitrary is None or \
                       distance > best_by_distance_arbitrary['distance']:
                        best_by_distance_arbitrary = result.copy()
                    
                    if is_zero_vortex:
                        if best_by_distance_zero_vortex is None or \
                           distance > best_by_distance_zero_vortex['distance']:
                            best_by_distance_zero_vortex = result.copy()
                
                # Update best by error rate
                if not np.isnan(logical_error_rate):
                    if best_by_error_rate_arbitrary is None or \
                       logical_error_rate < best_by_error_rate_arbitrary['logical_error_rate']:
                        best_by_error_rate_arbitrary = result.copy()
                    
                    if is_zero_vortex:
                        if best_by_error_rate_zero_vortex is None or \
                           logical_error_rate < best_by_error_rate_zero_vortex['logical_error_rate']:
                            best_by_error_rate_zero_vortex = result.copy()
                
                successful += 1
                
                # Print progress for notable results
                if is_standard and is_zero_vortex:
                    print(f"\nStandard order (zero vortex): distance={distance}, "
                          f"error_rate={logical_error_rate:.6f}")
                
            except Exception as e:
                print(f"Error testing sample {sample_idx}, vortex {vortex_counts}: {e}")
                failed += 1
                continue
        
        # Checkpoint: save results periodically
        if (sample_idx + 1) % checkpoint_interval == 0:
            _save_checkpoint(all_results, output_csv, result_fieldnames)
            _save_gate_order_summary(
                best_by_distance_arbitrary,
                best_by_distance_zero_vortex,
                best_by_error_rate_arbitrary,
                best_by_error_rate_zero_vortex,
                lattice_vectors,
                summary_csv
            )
    
    # Final save
    _save_checkpoint(all_results, output_csv, result_fieldnames)
    _save_gate_order_summary(
        best_by_distance_arbitrary,
        best_by_distance_zero_vortex,
        best_by_error_rate_arbitrary,
        best_by_error_rate_zero_vortex,
        lattice_vectors,
        summary_csv
    )
    
    # Print summary
    print(f"\n=== Search Complete ===")
    print(f"Total configurations tested: {total_tested}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    print(f"\n--- Best with Arbitrary Vortex ---")
    if best_by_distance_arbitrary:
        print(f"Best by distance:")
        print(f"  Distance: {best_by_distance_arbitrary['distance']}")
        print(f"  Error rate: {best_by_distance_arbitrary['logical_error_rate']:.6f}")
        print(f"  Vortex: [{best_by_distance_arbitrary['vortex_0']}, "
              f"{best_by_distance_arbitrary['vortex_1']}, "
              f"{best_by_distance_arbitrary['vortex_2']}, "
              f"{best_by_distance_arbitrary['vortex_3']}]")
        print(f"  Standard order: {best_by_distance_arbitrary['is_standard_order']}")
    
    if best_by_error_rate_arbitrary:
        print(f"Best by error rate:")
        print(f"  Distance: {best_by_error_rate_arbitrary['distance']}")
        print(f"  Error rate: {best_by_error_rate_arbitrary['logical_error_rate']:.6f}")
        print(f"  Vortex: [{best_by_error_rate_arbitrary['vortex_0']}, "
              f"{best_by_error_rate_arbitrary['vortex_1']}, "
              f"{best_by_error_rate_arbitrary['vortex_2']}, "
              f"{best_by_error_rate_arbitrary['vortex_3']}]")
        print(f"  Standard order: {best_by_error_rate_arbitrary['is_standard_order']}")
    
    print(f"\n--- Best with Zero Vortex ---")
    if best_by_distance_zero_vortex:
        print(f"Best by distance:")
        print(f"  Distance: {best_by_distance_zero_vortex['distance']}")
        print(f"  Error rate: {best_by_distance_zero_vortex['logical_error_rate']:.6f}")
        print(f"  Standard order: {best_by_distance_zero_vortex['is_standard_order']}")
    
    if best_by_error_rate_zero_vortex:
        print(f"Best by error rate:")
        print(f"  Distance: {best_by_error_rate_zero_vortex['distance']}")
        print(f"  Error rate: {best_by_error_rate_zero_vortex['logical_error_rate']:.6f}")
        print(f"  Standard order: {best_by_error_rate_zero_vortex['is_standard_order']}")
    
    print(f"\nDetailed results saved to: {output_csv}")
    print(f"Summary saved to: {summary_csv}")


def _save_checkpoint(results: List[Dict], output_csv: str, fieldnames: List[str]):
    """Save checkpoint of results to CSV."""
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def _save_gate_order_summary(
    best_by_distance_arbitrary: Optional[Dict],
    best_by_distance_zero_vortex: Optional[Dict],
    best_by_error_rate_arbitrary: Optional[Dict],
    best_by_error_rate_zero_vortex: Optional[Dict],
    lattice_vectors: List[List[int]],
    summary_csv: str
):
    """Save summary of best gate orders to CSV."""
    summary_fieldnames = [
        'metric_type', 'vortex_type',
        'lattice_vec_0', 'lattice_vec_1', 'lattice_vec_2', 'lattice_vec_3',
        'vortex_0', 'vortex_1', 'vortex_2', 'vortex_3',
        'permutation', 'gate_order_str',
        'n', 'k', 'distance', 'logical_error_rate', 'error_bar',
        'shots', 'errors', 'p_phenomenological', 'p_cx', 'is_standard_order'
    ]
    
    summary_rows = []
    
    def add_summary_row(result: Dict, metric_type: str, vortex_type: str):
        summary_rows.append({
            'metric_type': metric_type,
            'vortex_type': vortex_type,
            'lattice_vec_0': str(lattice_vectors[0]),
            'lattice_vec_1': str(lattice_vectors[1]),
            'lattice_vec_2': str(lattice_vectors[2]),
            'lattice_vec_3': str(lattice_vectors[3]),
            'vortex_0': result['vortex_0'],
            'vortex_1': result['vortex_1'],
            'vortex_2': result['vortex_2'],
            'vortex_3': result['vortex_3'],
            'permutation': result['permutation'],
            'gate_order_str': result['gate_order_str'],
            'n': result['n'],
            'k': result['k'],
            'distance': result['distance'],
            'logical_error_rate': result['logical_error_rate'],
            'error_bar': result['error_bar'],
            'shots': result['shots'],
            'errors': result['errors'],
            'p_phenomenological': result['p_phenomenological'],
            'p_cx': result['p_cx'],
            'is_standard_order': result['is_standard_order']
        })
    
    # Distance - arbitrary vortex
    if best_by_distance_arbitrary:
        add_summary_row(best_by_distance_arbitrary, 'distance', 'arbitrary')
    
    # Distance - zero vortex
    if best_by_distance_zero_vortex:
        add_summary_row(best_by_distance_zero_vortex, 'distance', 'zero_vortex')
    
    # Error rate - arbitrary vortex
    if best_by_error_rate_arbitrary:
        add_summary_row(best_by_error_rate_arbitrary, 'error_rate', 'arbitrary')
    
    # Error rate - zero vortex
    if best_by_error_rate_zero_vortex:
        add_summary_row(best_by_error_rate_zero_vortex, 'error_rate', 'zero_vortex')
    
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Search for optimal gate orders with random permutations'
    )
    parser.add_argument('--lattice', type=str, default='[[0,6,0,0], [6, 0, 0, 0], [1, 1, -1, 0], [1, 1, 0, -1]]',
                        help='Lattice vectors as nested list, e.g. "[[3,0,0,0],[0,3,0,0],[0,0,3,0],[0,0,0,3]]"')
    parser.add_argument('--p-phenom', type=float, default=0.0,
                        help='Phenomenological error rate [default: 0.0]')
    parser.add_argument('--p-cx', type=float, default=0.002,
                        help='CNOT gate error rate [default: 0.001]')
    parser.add_argument('--shots', type=int, default=100000,
                        help='Shots per configuration [default: 100000]')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of random gate order permutations to sample [default: 1000]')
    parser.add_argument('--workers', type=int, default=10,
                        help='Number of parallel workers [default: 10]')
    parser.add_argument('--max-errors', type=int, default=100,
                        help='Maximum errors before stopping [default: 100]')
    parser.add_argument('--noisy-cycles', type=int, default=4,
                        help='Number of noisy cycles [default: 4]')
    parser.add_argument('--noiseless-init', type=int, default=3,
                        help='Number of initial noiseless cycles [default: 3]')
    parser.add_argument('--noiseless-final', type=int, default=3,
                        help='Number of final noiseless cycles [default: 3]')
    parser.add_argument('--basis', type=str, default='Z', choices=['Z', 'X'],
                        help='Measurement basis [default: Z]')
    parser.add_argument('--output', type=str, default='gate_order_search_results.csv',
                        help='Output CSV for detailed results [default: gate_order_search_results.csv]')
    parser.add_argument('--summary', type=str, default='best_gate_orders_summary.csv',
                        help='Output CSV for best configurations [default: best_gate_orders_summary.csv]')
    parser.add_argument('--checkpoint', type=int, default=1,
                        help='Checkpoint interval [default: 50]')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility [default: None]')
    parser.add_argument('--no-standard', action='store_true',
                        help='Skip testing the standard BB gate order')
    
    args = parser.parse_args()
    
    # Parse lattice vectors
    lattice_vectors = ast.literal_eval(args.lattice)
    
    search_gate_orders(
        lattice_vectors=lattice_vectors,
        p_phenomenological=args.p_phenom,
        p_cx=args.p_cx,
        shots_per_point=args.shots,
        num_samples=args.num_samples,
        num_workers=args.workers,
        max_errors=args.max_errors,
        num_noisy_cycles=args.noisy_cycles,
        num_noiseless_cycles_init=args.noiseless_init,
        num_noiseless_cycles_final=args.noiseless_final,
        basis=args.basis,
        output_csv=args.output,
        summary_csv=args.summary,
        checkpoint_interval=args.checkpoint,
        random_seed=args.seed,
        include_standard=not args.no_standard
    )
