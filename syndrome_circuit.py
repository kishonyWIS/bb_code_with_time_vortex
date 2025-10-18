"""
Syndrome extraction circuit builder for Stim.
"""

import stim
import numpy as np
from typing import List, Union, Dict, Optional
from circuit_operations import CircuitOperation, Reset, Measure, CX, Depolarize2, Detector, Observable
from gate_order import GateOrder
from qubit_system import QubitSystem
from lattice import Point
from logical_operators import LogicalOperators


class SyndromeCircuit:
    """
    Builder for syndrome extraction circuits with time-ordered operations.
    """
    
    def __init__(self, qubit_system: QubitSystem, lattice_points: List[Point], 
                 gate_order: GateOrder, num_cycles: int = None, 
                 num_noisy_cycles: int = 0, p_cx: float = 0.0,
                 basis: str = 'Z', include_observables: bool = True, 
                 include_detectors: bool = True, vortex_counts: Optional[List[int]] = None):
        """
        Initialize syndrome circuit builder.
        
        Args:
            qubit_system: Qubit system for getting qubit indices
            lattice_points: List of lattice points to include in syndrome extraction
            gate_order: Ordering of CX gates
            num_cycles: Number of syndrome extraction cycles (deprecated, use num_noisy_cycles)
            num_noisy_cycles: Number of noisy cycles (total cycles = 2 + num_noisy_cycles)
            p_cx: Depolarizing error probability after CX gates
            basis: Basis for data qubit measurements ('Z' or 'X')
            include_observables: Whether to include observable instructions
            include_detectors: Whether to include detector instructions
            vortex_counts: List of vortex counts per lattice vector (length = lattice.dimension)
        """
        self.qubit_system = qubit_system
        self.lattice_points = lattice_points
        self.gate_order = gate_order
        self.basis = basis
        self.include_observables = include_observables
        self.include_detectors = include_detectors
        
        # Validate vortex_counts
        if vortex_counts is not None:
            if len(vortex_counts) != qubit_system.lattice.dimension:
                raise ValueError(f"vortex_counts length {len(vortex_counts)} must equal lattice dimension {qubit_system.lattice.dimension}")
        self.vortex_counts = vortex_counts
        
        # Handle backwards compatibility
        if num_cycles is not None and num_noisy_cycles == 0:
            # Old style: num_cycles specifies total cycles, all noise-free
            self.num_cycles = num_cycles
            self.num_noisy_cycles = 0
        else:
            # New style: 2 + num_noisy_cycles total cycles
            self.num_noisy_cycles = num_noisy_cycles
            self.num_cycles = 2 + num_noisy_cycles if num_noisy_cycles > 0 else (num_cycles or 1)
        
        self.p_cx = p_cx
        self._operations: List[CircuitOperation] = []
        self._measurement_indices: Dict[int, List[int]] = {}  # Maps qubit -> list of measurement indices
        self._next_measurement_index = 0
        
        # Compute logical operators if needed
        self.logical_operators: Optional[LogicalOperators] = None
        if include_observables:
            self.logical_operators = LogicalOperators(qubit_system, lattice_points, gate_order)
    
    def build_operations(self) -> List[CircuitOperation]:
        """
        Build the complete list of circuit operations for syndrome extraction.
        
        Uses staged construction:
        1. Build syndrome measurement cycles
        2. Apply vortex delays to syndrome cycles only
        3. Add initial state preparation (before syndrome cycles)
        4. Construct measurement indices from operations
        5. Add detectors (after syndrome cycles)
        6. Add final measurements
        7. Add observables
        
        Returns:
            List of all circuit operations in time order
        """
        self._operations = []
        
        # Stage 1: Build syndrome measurement cycles
        self._build_syndrome_cycles()
        
        # Populate positions for syndrome operations
        self._populate_operation_positions()
        
        # Apply vortex delays (only affects syndrome operations)
        if self.vortex_counts is not None:
            self._apply_vortex_delays()
        
        # Find time bounds of syndrome cycles
        min_time = min(op.time for op in self._operations)
        max_time = max(op.time for op in self._operations)
        
        # Stage 2: Add initial state preparation (before syndrome cycles)
        if self.basis == 'X':
            self._add_initial_state_preparation(min_time - 1.0)
        
        # Stage 3: Construct measurement indices from operations
        self._construct_measurement_indices()
        
        # Stage 4: Add detectors (after syndrome cycles)
        if self.include_detectors:
            self._add_detectors_after_circuit(max_time + 1.0)
        
        # Stage 5: Add final measurements
        final_measure_time = max_time + 2.0
        self._add_final_measurements(final_measure_time)
        
        # Stage 6: Add observables
        if self.include_observables and self.logical_operators is not None:
            self._add_observables(max_time + 3.0)
        
        return self._operations.copy()
    
    def _add_reset_operations(self, time: float) -> None:
        """Add reset operations for all ancillas at the given time."""
        for point in self.lattice_points:
            # Reset X ancillas in X basis
            x_ancilla = self.qubit_system.get_qubit_index(point, "X_anc")
            self._operations.append(Reset(time, x_ancilla, "X"))
            
            # Reset Z ancillas in Z basis
            z_ancilla = self.qubit_system.get_qubit_index(point, "Z_anc")
            self._operations.append(Reset(time, z_ancilla, "Z"))
    
    def _add_single_ancilla_measurement(self, time: float, ancilla_idx: int, basis: str) -> None:
        """Helper to measure a single ancilla (no index tracking)."""
        measure_op = Measure(time, ancilla_idx, basis)
        self._operations.append(measure_op)
    
    def _add_measure_operations(self, time: float) -> None:
        """Add measurement operations for all ancillas at the given time."""
        for point in self.lattice_points:
            # Measure X ancillas in X basis
            x_ancilla = self.qubit_system.get_qubit_index(point, "X_anc")
            self._add_single_ancilla_measurement(time, x_ancilla, "X")
            
            # Measure Z ancillas in Z basis
            z_ancilla = self.qubit_system.get_qubit_index(point, "Z_anc")
            self._add_single_ancilla_measurement(time, z_ancilla, "Z")
    
    def to_stim_circuit(self) -> stim.Circuit:
        """
        Convert the operations to a Stim circuit.
        
        Stim automatically batches compatible operations (e.g., consecutive RX operations).
        
        Returns:
            Stim circuit object
        """
        if not self._operations:
            self.build_operations()
        
        # Sort operations by time
        sorted_operations = sorted(self._operations, key=lambda op: op.time)
        
        # Create circuit and append operations
        # Stim automatically handles batching of compatible operations when using +=
        circuit = stim.Circuit()
        for op in sorted_operations:
            circuit += stim.Circuit(op.to_stim())
        
        return circuit
    
    def get_operations(self) -> List[CircuitOperation]:
        """Get the list of operations (builds if not already built)."""
        if not self._operations:
            self.build_operations()
        return self._operations.copy()
    
    def print_operations(self) -> None:
        """Print all operations with their times."""
        operations = self.get_operations()
        for op in operations:
            print(f"t={op.time}: {op.to_stim()}")
    
    def get_operation_summary(self) -> dict:
        """
        Get a summary of the circuit operations.
        
        Returns:
            Dictionary with operation counts and timing info
        """
        operations = self.get_operations()
        
        summary = {
            "total_operations": len(operations),
            "num_cycles": self.num_cycles,
            "num_lattice_points": len(self.lattice_points),
            "time_range": (min(op.time for op in operations), max(op.time for op in operations)),
            "operation_counts": {}
        }
        
        # Count operations by type
        for op in operations:
            op_type = type(op).__name__
            summary["operation_counts"][op_type] = summary["operation_counts"].get(op_type, 0) + 1
        
        return summary
    
    def _add_initial_state_preparation(self, time: float) -> None:
        """Add initial state preparation (RX for X-basis measurement)."""
        for point in self.lattice_points:
            l_qubit = self.qubit_system.get_qubit_index(point, 'L')
            r_qubit = self.qubit_system.get_qubit_index(point, 'R')
            # RX is the same as R in Stim (resets to |+> state)
            self._operations.append(Reset(time, l_qubit, 'X'))
            self._operations.append(Reset(time, r_qubit, 'X'))
    
    def _add_final_measurements(self, time: float) -> None:
        """Add final measurements of all data qubits."""
        for point in self.lattice_points:
            l_qubit = self.qubit_system.get_qubit_index(point, 'L')
            r_qubit = self.qubit_system.get_qubit_index(point, 'R')
            
            # Measure in the specified basis
            l_measure = Measure(time, l_qubit, self.basis)
            r_measure = Measure(time, r_qubit, self.basis)
            
            self._operations.append(l_measure)
            self._operations.append(r_measure)
    
    def _add_observables_for_basis(self, time: float, logical_ops: List[np.ndarray], observable_id_offset: int) -> None:
        """Helper to add observables for a specific basis."""
        for i, logical_op in enumerate(logical_ops):
            measurement_indices = []
            for j, val in enumerate(logical_op):
                if val == 1:
                    # Calculate the measurement index relative to the end
                    # Data qubits are measured at the end in order
                    measurement_indices.append(len(self.lattice_points) * 2 - j)
            if measurement_indices:
                obs = Observable(time, observable_id_offset + i, measurement_indices)
                self._operations.append(obs)
    
    def _add_observables(self, time: float) -> None:
        """Add observable instructions for logical operators."""
        if self.logical_operators is None:
            return
        
        num_logical = self.logical_operators.get_num_logical_qubits()
        
        if self.basis == 'Z':
            # Z-basis measurements detect Z logical operators
            ops = self.logical_operators.logical_z_ops[:num_logical]
            self._add_observables_for_basis(time, ops, 0)
        else:  # X basis
            # X-basis measurements detect X logical operators
            ops = self.logical_operators.logical_x_ops[:num_logical]
            self._add_observables_for_basis(time, ops, 0)
    
    def _add_detectors_after_circuit(self, time: float) -> None:
        """
        Add detectors after circuit construction.
        
        For each ancilla qubit, track measurement indices across cycles.
        Create detectors for consecutive measurements of the same ancilla.
        """
        # Create detectors for consecutive measurements of each ancilla
        for ancilla_idx, measurement_indices in self._measurement_indices.items():
            # For each ancilla, create detectors between consecutive measurements
            for i in range(len(measurement_indices) - 1):
                # Detector compares consecutive measurements
                # rec[-k] where k is the distance from the most recent measurement
                recent_idx = measurement_indices[i + 1]  # More recent measurement
                previous_idx = measurement_indices[i]    # Previous measurement
                
                # Calculate relative indices from the end
                # At the time of detector creation, the most recent measurement is at self._next_measurement_index - 1
                recent_relative = self._next_measurement_index - recent_idx
                previous_relative = self._next_measurement_index - previous_idx
                
                detector_op = Detector(
                    time, 
                    ancilla_idx,  # Dummy qubit index (not used in Stim)
                    previous_relative, 
                    recent_relative
                )
                self._operations.append(detector_op)
    
    def _populate_operation_positions(self) -> None:
        """
        Populate the position field for all operations based on their affected qubits.
        Operations without spatial position (Observable) are left as None.
        """
        for operation in self._operations:
            affected_qubits = operation.affected_qubits()
            if not affected_qubits:
                # Operations like Observable don't have spatial position
                continue
            
            # Get positions of all affected qubits
            qubit_positions = []
            for qubit_idx in affected_qubits:
                position = self.qubit_system.get_qubit_position(qubit_idx)
                if position is not None:
                    qubit_positions.append(position)
            
            if qubit_positions:
                # Compute periodic mean of qubit positions
                mean_position = self.qubit_system.lattice.compute_periodic_mean(qubit_positions)
                operation.position = mean_position.coords
    
    def _build_syndrome_cycles(self) -> None:
        """
        Build syndrome measurement cycles with proper timing.
        
        Uses new timing scheme where each cycle has duration 1.0:
        - Total steps per cycle: s = len(gate_order.descriptors) + 2
        - Time step size: dt = 1.0 / s
        - Cycle i: reset at i, CX gates at i + k*dt, measurements at i + (s-1)*dt
        
        For noisy circuits:
        - First cycle (0): noise-free
        - Middle cycles (1 to num_noisy_cycles): noisy
        - Last cycle (num_noisy_cycles + 1): noise-free
        """
        # Calculate timing parameters
        num_cx_layers = len(self.gate_order.descriptors)
        steps_per_cycle = num_cx_layers + 2  # reset + CX gates + measurements
        dt = 1.0 / steps_per_cycle
        
        for cycle in range(self.num_cycles):
            cycle_start_time = float(cycle)
            
            # Determine if this cycle should be noisy
            is_noisy = False
            if self.num_noisy_cycles > 0 and self.p_cx > 0:
                # First cycle (0) is noise-free
                # Last cycle (num_cycles - 1) is noise-free
                # Middle cycles are noisy
                is_noisy = (cycle > 0 and cycle < self.num_cycles - 1)
            
            # Reset all ancillas at cycle start
            self._add_reset_operations(cycle_start_time)
            
            # Apply CX gates with proper timing - one descriptor (layer) at a time
            for cx_step, descriptor in enumerate(self.gate_order.descriptors):
                cx_time = cycle_start_time + (cx_step + 1) * dt
                cx_operations = descriptor.to_operations(
                    self.qubit_system, self.lattice_points, cx_time
                )
                self._operations.extend(cx_operations)
                
                # Add noise after CX gates if this is a noisy cycle
                if is_noisy:
                    noise_time = cx_time + 1e-9  # Small epsilon after CX gate
                    for cx_op in cx_operations:
                        noise_op = Depolarize2(noise_time, cx_op.control, cx_op.target, self.p_cx)
                        self._operations.append(noise_op)
            
            # Measure all ancillas at end of cycle
            measure_time = cycle_start_time + (steps_per_cycle - 1) * dt
            self._add_measure_operations(measure_time)
    
    def _construct_measurement_indices(self) -> None:
        """
        Build _measurement_indices dictionary by scanning operations.
        
        Sorts operations by time first, then tracks measurement indices
        for each qubit. This ensures indices are correct after vortex delays.
        """
        # Sort operations by time
        sorted_ops = sorted(self._operations, key=lambda op: op.time)
        
        # Track measurement indices
        self._measurement_indices = {}
        measurement_index = 0
        
        for op in sorted_ops:
            if isinstance(op, Measure):
                qubit = op.qubit
                if qubit not in self._measurement_indices:
                    self._measurement_indices[qubit] = []
                self._measurement_indices[qubit].append(measurement_index)
                measurement_index += 1
        
        # Update next measurement index for future operations
        self._next_measurement_index = measurement_index

    def _apply_vortex_delays(self) -> None:
        """
        Apply time delays to operations based on their position and vortex configuration.
        Delay = dot_product(lattice_coords, vortex_counts) where lattice_coords
        are the normalized coordinates of the operation position in lattice basis.
        """
        if self.vortex_counts is None:
            return
        
        for operation in self._operations:
            if operation.position is None:
                # Skip operations without spatial position
                continue
            
            # Ensure position is a proper numpy array
            position = np.array(operation.position)
            if position.ndim == 0:
                # Skip scalar positions
                continue
            
            # Convert position to lattice basis coordinates
            lattice_coords = self.qubit_system.lattice.lattice_matrix_inv @ position
            
            # Compute delay as dot product with vortex counts
            delay = np.dot(lattice_coords, self.vortex_counts)
            
            # Apply delay to operation time
            operation.time += delay
