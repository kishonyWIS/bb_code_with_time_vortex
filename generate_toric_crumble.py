#!/usr/bin/env python3
"""
Script to generate a Crumble URL for visualizing a toric code circuit.
Configure the parameters below to customize the circuit.
"""

import numpy as np
from lattice import Lattice
from qubit_system import QubitSystem
from gate_order import GateOrder, GateDescriptor
from syndrome_circuit import SyndromeCircuit
import urllib.parse
import stim

# =============================================================================
# CONFIGURATION - Modify these parameters to customize your circuit
# =============================================================================

# Number of noisy syndrome extraction cycles (total cycles = num_noiseless_cycles_init + num_noisy_cycles + num_noiseless_cycles_final)
NUM_NOISY_CYCLES = 2
NUM_NOISLESS_CYCLES_INIT = 6
NUM_NOISLESS_CYCLES_FINAL = 6

# Depolarizing error probability after CX gates
P_CX = 0.001

# Measurement basis ('Z' or 'X')
BASIS = 'Z'

# Time vortex configuration [vortex_count_x, vortex_count_y]
# Set to [0, 0] for no vortices, [1, 0] for one vortex in x-direction, etc.
VORTEX_COUNTS = [1, 0]
LATTICE_VECTORS = [[4, 0], [0, 5]]

# Whether to include observable instructions
INCLUDE_OBSERVABLES = True

# Whether to analyze logical errors and distances
ANALYZE_ERRORS = True

# =============================================================================

def generate_toric_code_crumble_url():
    """
    Generate a Crumble URL for the configured toric code circuit.
    
    Returns:
        Crumble URL string and circuit information
    """
    
    # Create lattice
    lattice = Lattice(LATTICE_VECTORS)
    qsys = QubitSystem(lattice)
    points = lattice.get_all_lattice_points()
    
    print(f"=== Toric Code Configuration ===")
    print(f"Lattice vectors: {LATTICE_VECTORS}")
    print(f"Number of lattice points: {len(points)}")
    print(f"Total qubits: {len(points) * 4}")
    print(f"Noisy cycles: {NUM_NOISY_CYCLES} (total cycles: {2 + NUM_NOISY_CYCLES})")
    print(f"Noise probability: {P_CX}")
    print(f"Basis: {BASIS}")
    print(f"Vortex counts: {VORTEX_COUNTS}")
    print(f"Observables: {INCLUDE_OBSERVABLES}")
    print(f"Error analysis: {ANALYZE_ERRORS}")
    
    # # Create gate order
    # gate_order = GateOrder([
    #     GateDescriptor('Z', 'on_site_L'), GateDescriptor('Z', 'axis_1'), 
    #     GateDescriptor('Z', 'on_site_R'), GateDescriptor('Z', 'axis_0'),
    #     GateDescriptor('X', 'on_site_L'), GateDescriptor('X', 'axis_1'), 
    #     GateDescriptor('X', 'on_site_R'), GateDescriptor('X', 'axis_0')
    # ])
    # Create gate order
    gate_order = GateOrder([
        GateDescriptor('X', 'on_site_R'), GateDescriptor('Z', 'axis_0'),
        GateDescriptor('X', 'axis_1'), GateDescriptor('Z', 'on_site_R'),
        GateDescriptor('X', 'on_site_L'), GateDescriptor('Z', 'axis_1'),
        GateDescriptor('X', 'axis_0'), GateDescriptor('Z', 'on_site_L'),
    ])


    # Create circuit
    circuit = SyndromeCircuit(
        qsys, points, gate_order, 
        num_noisy_cycles=NUM_NOISY_CYCLES,
        num_noiseless_cycles_init=NUM_NOISLESS_CYCLES_INIT,
        num_noiseless_cycles_final=NUM_NOISLESS_CYCLES_FINAL,
        p_cx=P_CX,
        basis=BASIS, 
        include_observables=INCLUDE_OBSERVABLES,
        vortex_counts=VORTEX_COUNTS
    )
    
    # Generate Stim circuit
    stim_circuit = circuit.to_stim_circuit()
    
    # Convert to string and create Crumble URL
    circuit_str = str(stim_circuit)
    encoded_circuit = urllib.parse.quote(circuit_str)
    crumble_url = f"https://algassert.com/crumble#circuit={encoded_circuit}"
    
    # Also generate a 3D matching graph HTML
    try:
        diag = stim_circuit.diagram('matchgraph-3d-html')
        output_path = 'matchgraph_3d.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(str(diag))
        print(f"\nWrote 3D match graph to {output_path}")
    except Exception as e:
        print(f"\nFailed to generate 3D match graph: {e}")
    
    return crumble_url, circuit_str, circuit

def main():
    """Generate and display the Crumble URL."""

    
    print("=== Toric Code Crumble URL Generator ===\n")
    
    # Generate the circuit
    url, circuit_str, circuit = generate_toric_code_crumble_url()
    
    # Generate Stim circuit for error analysis
    stim_circuit = circuit.to_stim_circuit()

        # Show circuit structure
    print(f"\n=== Circuit Structure ===")
    print(stim_circuit)
    
    print(f"\n=== Generated Crumble URL ===")
    print(url)
    
    # Show timing information
    print(f"\n=== Timing Information ===")
    ops = sorted(circuit.get_operations(), key=lambda o: o.time)
    print(f"Total operations: {len(ops)}")
    print(f"Time range: [{min(op.time for op in ops):.3f}, {max(op.time for op in ops):.3f}]")
    
    print(f"\nFirst 10 operations:")
    for i, op in enumerate(ops[:10]):
        print(f"  {i+1}: t={op.time:.3f} {type(op).__name__}")
    
    if len(ops) > 10:
        print(f"  ... and {len(ops) - 10} more operations")
    
    # Show vortex effect
    if VORTEX_COUNTS != [0, 0]:
        print(f"\n=== Vortex Effect ===")
        print("Operations are spread out in time based on spatial position due to vortices.")
        
        # Count operations by time
        time_counts = {}
        for op in ops:
            time_counts[op.time] = time_counts.get(op.time, 0) + 1
        
        print(f"Number of distinct time steps: {len(time_counts)}")
        print(f"Time steps: {sorted(time_counts.keys())}")
    else:
        print(f"\n=== No Vortices ===")
        print("All operations at the same time step are grouped together.")

    # Show error analysis
    if ANALYZE_ERRORS and INCLUDE_OBSERVABLES:
        print(f"\n=== Error Analysis ===")
        print(f"Number of logical qubits: {circuit.logical_operators.get_num_logical_qubits()}")
        
        try:
            # Get shortest graphlike error
            shortest_error = stim_circuit.shortest_graphlike_error(canonicalize_circuit_errors=True)
            print(f"Shortest graphlike error: {len(shortest_error)}")
            # explain_explanations(stim_circuit, shortest_error)
            # Search for minimal undetectable logical errors
            minimal_error = stim_circuit.search_for_undetectable_logical_errors(
                dont_explore_detection_event_sets_with_size_above=9999,
                dont_explore_edges_with_degree_above=9999,
                dont_explore_edges_increasing_symptom_degree=False,
                canonicalize_circuit_errors=True
            )
            print(f"Minimal found error: {len(minimal_error)}")
            # explain_explanations(stim_circuit, minimal_error)
        except ValueError as e:
            print(f"Error analysis failed: {str(e)}")
            print("This is likely due to non-deterministic observables (X-basis measurements with Z-basis observables)")
    elif ANALYZE_ERRORS and not INCLUDE_OBSERVABLES:
        print(f"\n=== Error Analysis ===")
        print("Error analysis requires observables to be enabled (INCLUDE_OBSERVABLES = True)")

def explain_explanations(c:stim.Circuit, explanations):
    for i, ex in enumerate(explanations):
        print(f"---- Explanation #{i} ----")
        # Which DEM terms (detectors/logicals) the shortest error hits:
        for term in ex.dem_error_terms:
            print("DEM term:", term)
        # Where in the circuit those terms came from:
        for loc in ex.circuit_error_locations:
            print("At instruction index:", loc.stack_frames[-1].instruction_offset,
                "gate:", c[loc.stack_frames[-1].instruction_offset].name,
                "targets:", c[loc.stack_frames[-1].instruction_offset].targets_copy())

if __name__ == "__main__":
    main()
