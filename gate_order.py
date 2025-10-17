"""
Gate order specification for syndrome extraction circuits.
"""

from typing import List, Tuple
from dataclasses import dataclass
from circuit_operations import CX
from qubit_system import QubitSystem
from lattice import Point


@dataclass
class GateDescriptor:
    """
    Describes a type of CX gate connection in the syndrome extraction circuit.
    """
    ancilla_type: str  # "X" or "Z"
    connection_type: str  # e.g., "on_site_L", "on_site_R", "axis_0", "axis_1", etc.
    
    def get_axis_from_connection_type(self) -> int:
        """
        Extract axis number from connection type like "axis_0", "axis_1", etc.
        Returns -1 for on_site connections.
        """
        if self.connection_type.startswith("axis_"):
            try:
                return int(self.connection_type.split("_")[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid axis connection type: {self.connection_type}")
        return -1
    
    def get_qubit_type_from_connection_type(self) -> str:
        """
        Extract qubit type (L or R) from connection type.
        """
        if self.connection_type.startswith("on_site_"):
            return self.connection_type.split("_")[2]  # "on_site_L" -> "L"
        elif self.connection_type.startswith("axis_"):
            # For axis connections, we need to determine L or R based on ancilla type and axis
            # This will be handled in the GateOrder class
            return None  # Will be determined dynamically
        else:
            raise ValueError(f"Invalid connection type: {self.connection_type}")
    
    def get_data_qubit_label(self, axis: int) -> str:
        """
        Get the data qubit label (L or R) for a given axis based on stabilizer type.
        
        Rules:
        - X stabilizers: even axes → R, odd axes → L
        - Z stabilizers: even axes → L, odd axes → R
        
        Args:
            axis: Axis number (0-based)
            
        Returns:
            'L' or 'R'
        """
        if self.ancilla_type == "X":
            return "R" if axis % 2 == 0 else "L"
        elif self.ancilla_type == "Z":
            return "L" if axis % 2 == 0 else "R"
        else:
            raise ValueError(f"Invalid ancilla type: {self.ancilla_type}")
    
    def get_shift_direction(self, axis: int) -> int:
        """
        Get the shift direction (+1 or -1) for a given axis based on stabilizer type.
        
        Rules:
        - X stabilizers: positive shift (+1)
        - Z stabilizers: negative shift (-1)
        
        Args:
            axis: Axis number (0-based, not used for direction but kept for consistency)
            
        Returns:
            +1 for positive shift, -1 for negative shift
        """
        if self.ancilla_type == "X":
            return 1  # positive shift
        elif self.ancilla_type == "Z":
            return -1  # negative shift
        else:
            raise ValueError(f"Invalid ancilla type: {self.ancilla_type}")


class GateOrder:
    """
    Manages the ordering of CX gates in syndrome extraction circuits.
    """
    
    def __init__(self, descriptors: List[GateDescriptor]):
        """
        Initialize with a list of gate descriptors.
        
        Args:
            descriptors: Ordered list of gate types to apply
        """
        self.descriptors = descriptors
    
    @staticmethod
    def get_default_order(dimension: int) -> 'GateOrder':
        """
        Get the default gate order for any dimension.
        
        Default order:
        - All Z ancilla gates: on_site_L, on_site_R, axis_0, axis_1, ..., axis_(D-1)
        - All X ancilla gates: on_site_L, on_site_R, axis_0, axis_1, ..., axis_(D-1)
        
        Args:
            dimension: Lattice dimension
            
        Returns:
            GateOrder with default ordering
        """
        descriptors = []
        
        # Z ancilla gates
        descriptors.append(GateDescriptor("Z", "on_site_L"))
        descriptors.append(GateDescriptor("Z", "on_site_R"))
        for axis in range(dimension):
            descriptors.append(GateDescriptor("Z", f"axis_{axis}"))
        
        # X ancilla gates
        descriptors.append(GateDescriptor("X", "on_site_L"))
        descriptors.append(GateDescriptor("X", "on_site_R"))
        for axis in range(dimension):
            descriptors.append(GateDescriptor("X", f"axis_{axis}"))
        
        return GateOrder(descriptors)
    
    def to_operations(self, qubit_system: QubitSystem, lattice_points: List[Point], 
                     base_time: float) -> List[CX]:
        """
        Generate CX operations according to this gate order.
        
        Args:
            qubit_system: Qubit system for getting qubit indices
            lattice_points: List of lattice points to apply gates to
            base_time: Starting time for the operations
            
        Returns:
            List of CX operations
        """
        operations = []
        current_time = base_time
        
        for descriptor in self.descriptors:
            for point in lattice_points:
                cx_ops = self._create_cx_operations_for_descriptor(
                    qubit_system, point, descriptor, current_time
                )
                operations.extend(cx_ops)
            current_time += 1.0  # Each gate type gets its own time step
        
        return operations
    
    def _create_cx_operations_for_descriptor(self, qubit_system: QubitSystem, 
                                           point: Point, 
                                           descriptor: GateDescriptor, 
                                           time: float) -> List[CX]:
        """
        Create CX operations for a specific gate descriptor at a specific point.
        
        Args:
            qubit_system: Qubit system for getting qubit indices
            point: Lattice point
            descriptor: Gate descriptor
            time: Time for the operations
            
        Returns:
            List of CX operations
        """
        operations = []
        
        if descriptor.ancilla_type == "X":
            ancilla_qubit = qubit_system.get_qubit_index(point, "X_anc")
        else:  # Z
            ancilla_qubit = qubit_system.get_qubit_index(point, "Z_anc")
        
        if descriptor.connection_type.startswith("on_site_"):
            # On-site connection - only connect to qubits at the current point
            qubit_type = descriptor.get_qubit_type_from_connection_type()
            
            # Get the on-site qubit of the specified type
            data_qubit = qubit_system.get_qubit_index(point, qubit_type)
            
        elif descriptor.connection_type.startswith("axis_"):
            # Axis connection - use GateDescriptor methods to determine qubit and direction
            axis = descriptor.get_axis_from_connection_type()
            if axis < 0:
                raise ValueError(f"Invalid axis in connection type: {descriptor.connection_type}")
            
            # Get the data qubit label and shift direction from the descriptor
            data_qubit_label = descriptor.get_data_qubit_label(axis)
            shift_direction = descriptor.get_shift_direction(axis)
            
            # Calculate the shifted point
            shift = shift_direction * qubit_system.lattice.get_axis_vector(axis)
            shifted_point = qubit_system.lattice.get_shifted_point(point, shift)
            
            # Get the data qubit at the shifted point
            data_qubit = qubit_system.get_qubit_index(shifted_point, data_qubit_label)
        
        else:
            raise ValueError(f"Invalid connection type: {descriptor.connection_type}")
        
        # Create CX gate with consistent direction logic
        # For X ancillas: control=ancilla, target=data
        # For Z ancillas: control=data, target=ancilla
        if descriptor.ancilla_type == "X":
            operations.append(CX(time, ancilla_qubit, data_qubit))
        else:  # Z
            operations.append(CX(time, data_qubit, ancilla_qubit))
        
        return operations
