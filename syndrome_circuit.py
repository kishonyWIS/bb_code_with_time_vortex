"""
Syndrome extraction circuit builder for Stim.
"""

import stim
from typing import List, Union, Dict, Optional
from circuit_operations import CircuitOperation, Reset, Measure, Tick, CX, Depolarize2, Detector, Observable
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
                 include_detectors: bool = True):
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
        """
        self.qubit_system = qubit_system
        self.lattice_points = lattice_points
        self.gate_order = gate_order
        self.basis = basis
        self.include_observables = include_observables
        self.include_detectors = include_detectors
        
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
            self.logical_operators = LogicalOperators(qubit_system, lattice_points)
    
    def build_operations(self) -> List[CircuitOperation]:
        """
        Build the complete list of circuit operations for syndrome extraction.
        
        For noisy circuits:
        - First cycle (0): noise-free
        - Middle cycles (1 to num_noisy_cycles): noisy
        - Last cycle (num_noisy_cycles + 1): noise-free
        
        Adds final measurements and observables at the end.
        
        Returns:
            List of all circuit operations in time order
        """
        self._operations = []
        current_time = 0.0
        
        # Optional: Add initial state preparation for X-basis measurement
        if self.basis == 'X':
            self._add_initial_state_preparation(current_time)
            current_time += 1.0
        
        for cycle in range(self.num_cycles):
            cycle_start_time = current_time
            
            # Determine if this cycle should be noisy
            is_noisy = False
            if self.num_noisy_cycles > 0 and self.p_cx > 0:
                # First cycle (0) is noise-free
                # Last cycle (num_cycles - 1) is noise-free
                # Middle cycles are noisy
                is_noisy = (cycle > 0 and cycle < self.num_cycles - 1)
            
            # Reset all ancillas
            self._add_reset_operations(cycle_start_time)
            current_time += 1.0
            
            # Apply CX gates according to gate order
            cx_operations = self.gate_order.to_operations(
                self.qubit_system, self.lattice_points, current_time
            )
            self._operations.extend(cx_operations)
            
            # Add noise after CX gates if this is a noisy cycle
            if is_noisy:
                for cx_op in cx_operations:
                    noise_op = Depolarize2(cx_op.time, cx_op.control, cx_op.target, self.p_cx)
                    self._operations.append(noise_op)
            
            # Update time to after all CX gates
            if cx_operations:
                current_time = max(op.time for op in cx_operations) + 1.0
            else:
                current_time += 1.0
            
            # Measure all ancillas
            self._add_measure_operations(current_time)
            current_time += 1.0
            
            # Add TICK instruction between cycles (except after last cycle)
            if cycle < self.num_cycles - 1:
                self._operations.append(Tick(current_time))
                current_time += 1.0
        
        # Add detectors after last cycle but before final measurements
        if self.include_detectors:
            self._add_detectors_after_circuit(current_time)
            current_time += 1.0
        
        # Add final data qubit measurements
        self._add_final_measurements(current_time)
        current_time += 1.0
        
        # Add observables if requested
        if self.include_observables and self.logical_operators is not None:
            self._add_observables(current_time)
        
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
    
    def _add_measure_operations(self, time: float) -> None:
        """Add measurement operations for all ancillas at the given time."""
        for point in self.lattice_points:
            # Measure X ancillas in X basis
            x_ancilla = self.qubit_system.get_qubit_index(point, "X_anc")
            measure_op = Measure(time, x_ancilla, "X")
            self._operations.append(measure_op)
            
            # Detectors will be added after circuit construction
            
            # Update measurement index for this qubit
            if x_ancilla not in self._measurement_indices:
                self._measurement_indices[x_ancilla] = []
            self._measurement_indices[x_ancilla].append(self._next_measurement_index)
            self._next_measurement_index += 1
            
            # Measure Z ancillas in Z basis
            z_ancilla = self.qubit_system.get_qubit_index(point, "Z_anc")
            measure_op = Measure(time, z_ancilla, "Z")
            self._operations.append(measure_op)
            
            # Detectors will be added after circuit construction
            
            # Update measurement index for this qubit
            if z_ancilla not in self._measurement_indices:
                self._measurement_indices[z_ancilla] = []
            self._measurement_indices[z_ancilla].append(self._next_measurement_index)
            self._next_measurement_index += 1
    
    def to_stim_circuit(self) -> stim.Circuit:
        """
        Convert the operations to a Stim circuit.
        
        Returns:
            Stim circuit object
        """
        if not self._operations:
            self.build_operations()
        
        # Sort operations by time
        sorted_operations = sorted(self._operations, key=lambda op: op.time)
        
        # Group operations by time
        time_groups = {}
        for op in sorted_operations:
            if op.time not in time_groups:
                time_groups[op.time] = []
            time_groups[op.time].append(op)
        
        # Build Stim circuit
        stim_instructions = []
        for time in sorted(time_groups.keys()):
            operations_at_time = time_groups[time]
            
            # Group operations by type for efficiency
            resets = [op for op in operations_at_time if isinstance(op, Reset)]
            measures = [op for op in operations_at_time if isinstance(op, Measure)]
            detectors = [op for op in operations_at_time if isinstance(op, Detector)]
            observables = [op for op in operations_at_time if isinstance(op, Observable)]
            cx_gates = [op for op in operations_at_time if isinstance(op, CX)]
            depolarize_ops = [op for op in operations_at_time if isinstance(op, Depolarize2)]
            ticks = [op for op in operations_at_time if isinstance(op, Tick)]
            
            # Add operations in order: resets, CX gates + noise, measurements + detectors + observables, ticks
            if resets:
                # Group resets by basis
                x_resets = [op for op in resets if op.basis == "X"]
                z_resets = [op for op in resets if op.basis == "Z"]
                
                if x_resets:
                    qubits = [op.qubit for op in x_resets]
                    stim_instructions.append(f"RX {' '.join(map(str, qubits))}")
                
                if z_resets:
                    qubits = [op.qubit for op in z_resets]
                    stim_instructions.append(f"R {' '.join(map(str, qubits))}")
            
            if cx_gates:
                for cx in cx_gates:
                    stim_instructions.append(cx.to_stim())
            
            if depolarize_ops:
                for depol in depolarize_ops:
                    stim_instructions.append(depol.to_stim())
            
            if measures:
                # Group measurements by basis
                x_measures = [op for op in measures if op.basis == "X"]
                z_measures = [op for op in measures if op.basis == "Z"]
                
                if x_measures:
                    qubits = [op.qubit for op in x_measures]
                    stim_instructions.append(f"MX {' '.join(map(str, qubits))}")
                
                if z_measures:
                    qubits = [op.qubit for op in z_measures]
                    stim_instructions.append(f"M {' '.join(map(str, qubits))}")
            
            if detectors:
                for detector in detectors:
                    stim_instructions.append(detector.to_stim())
            
            if observables:
                for observable in observables:
                    stim_instructions.append(observable.to_stim())
            
            if ticks:
                stim_instructions.append("TICK")
        
        # Create Stim circuit
        circuit_str = "\n".join(stim_instructions)
        return stim.Circuit(circuit_str)
    
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
        from circuit_operations import Reset
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
            
            # Track measurement indices for observables
            self._next_measurement_index += 2
    
    def _add_observables(self, time: float) -> None:
        """Add observable instructions for logical operators."""
        if self.logical_operators is None:
            return
        
        num_logical = self.logical_operators.get_num_logical_qubits()
        
        # Add observables based on the basis
        if self.basis == 'Z':
            # Z-basis measurements detect Z logical operators
            for i in range(min(num_logical, len(self.logical_operators.logical_z_ops))):
                logical_op = self.logical_operators.logical_z_ops[i]
                # Find which data qubits have support
                measurement_indices = []
                for j, val in enumerate(logical_op):
                    if val == 1:
                        # Calculate the measurement index relative to the end
                        # Data qubits are measured at the end in order
                        measurement_indices.append(len(self.lattice_points) * 2 - j)
                
                if measurement_indices:
                    obs = Observable(time, i, measurement_indices)
                    self._operations.append(obs)
        else:  # X basis
            # X-basis measurements detect X logical operators
            for i in range(min(num_logical, len(self.logical_operators.logical_x_ops))):
                logical_op = self.logical_operators.logical_x_ops[i]
                # Find which data qubits have support
                measurement_indices = []
                for j, val in enumerate(logical_op):
                    if val == 1:
                        # Calculate the measurement index relative to the end
                        measurement_indices.append(len(self.lattice_points) * 2 - j)
                
                if measurement_indices:
                    obs = Observable(time, i, measurement_indices)
                    self._operations.append(obs)
    
    def _add_detectors_after_circuit(self, time: float) -> None:
        """
        Add detectors after circuit construction.
        
        For each ancilla qubit, track measurement indices across cycles.
        Create detectors for consecutive measurements of the same ancilla.
        """
        from circuit_operations import Detector
        
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
