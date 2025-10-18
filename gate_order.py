"""
Gate order specification for syndrome extraction circuits.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from circuit_operations import CX
from qubit_system import QubitSystem
from lattice import Point


class AncillaType(Enum):
    """Type of ancilla qubit."""
    X = "X"
    Z = "Z"


class ConnectionKind(Enum):
    """Kind of connection between ancilla and data qubit."""
    ON_SITE = "on_site"
    AXIS = "axis"


@dataclass
class GateDescriptor:
    """
    Describes a type of CX gate connection in the syndrome extraction circuit.
    """
    ancilla_type: str  # "X" or "Z" - kept for backward compatibility
    connection_type: str  # e.g., "on_site_L", "on_site_R", "axis_0", "axis_1", etc.
    
    # Explicit fields parsed from connection_type
    connection_kind: Optional[ConnectionKind] = None
    axis: Optional[int] = None
    data_qubit_label: Optional[str] = None
    
    def __post_init__(self):
        """Parse connection_type and populate explicit fields."""
        # Validate ancilla_type
        try:
            self.ancilla_type_enum = AncillaType(self.ancilla_type)
        except ValueError:
            raise ValueError(f"Invalid ancilla_type '{self.ancilla_type}'. Must be 'X' or 'Z'")
        
        # Parse connection_type
        if self.connection_type.startswith("on_site_"):
            self.connection_kind = ConnectionKind.ON_SITE
            self.data_qubit_label = self.connection_type.split("_")[2]  # "on_site_L" -> "L"
            if self.data_qubit_label not in ["L", "R"]:
                raise ValueError(f"Invalid data qubit label '{self.data_qubit_label}' in {self.connection_type}")
            self.axis = None
            
        elif self.connection_type.startswith("axis_"):
            self.connection_kind = ConnectionKind.AXIS
            try:
                self.axis = int(self.connection_type.split("_")[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid axis connection type: {self.connection_type}")
            self.data_qubit_label = None  # Will be determined dynamically
            
        else:
            raise ValueError(f"Invalid connection type: {self.connection_type}")
    
    def get_connected_data_qubit(self, qubit_system: QubitSystem, point: Point) -> Tuple[int, str]:
        """
        Get the data qubit index and label that this descriptor connects to.
        
        Args:
            qubit_system: Qubit system for getting qubit indices
            point: Lattice point for the ancilla
            
        Returns:
            Tuple of (qubit_index, qubit_label)
        """
        if self.connection_kind == ConnectionKind.ON_SITE:
            # On-site connection - data qubit is at the same point
            qubit_index = qubit_system.get_qubit_index(point, self.data_qubit_label)
            return qubit_index, self.data_qubit_label
            
        elif self.connection_kind == ConnectionKind.AXIS:
            # Axis connection - data qubit is at a shifted point
            data_qubit_label = self.get_data_qubit_label(self.axis)
            shift_direction = self.get_shift_direction(self.axis)
            
            shift = shift_direction * qubit_system.lattice.get_axis_vector(self.axis)
            shifted_point = qubit_system.lattice.get_shifted_point(point, shift)
            qubit_index = qubit_system.get_qubit_index(shifted_point, data_qubit_label)
            
            return qubit_index, data_qubit_label
            
        else:
            raise ValueError(f"Invalid connection kind: {self.connection_kind}")
    
    def get_axis_from_connection_type(self) -> int:
        """
        Extract axis number from connection type like "axis_0", "axis_1", etc.
        Returns -1 for on_site connections.
        """
        if self.connection_kind == ConnectionKind.AXIS:
            return self.axis
        return -1
    
    def get_qubit_type_from_connection_type(self) -> str:
        """
        Extract qubit type (L or R) from connection type.
        """
        if self.connection_kind == ConnectionKind.ON_SITE:
            return self.data_qubit_label
        elif self.connection_kind == ConnectionKind.AXIS:
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
        if self.ancilla_type_enum == AncillaType.X:
            return "R" if axis % 2 == 0 else "L"
        elif self.ancilla_type_enum == AncillaType.Z:
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
        if self.ancilla_type_enum == AncillaType.X:
            return 1  # positive shift
        elif self.ancilla_type_enum == AncillaType.Z:
            return -1  # negative shift
        else:
            raise ValueError(f"Invalid ancilla type: {self.ancilla_type}")
    
    def to_operations(self, qubit_system: QubitSystem, lattice_points: List[Point], 
                     time: float) -> List[CX]:
        """
        Generate CX operations for this descriptor across all lattice points.
        
        Args:
            qubit_system: Qubit system for getting qubit indices
            lattice_points: List of lattice points to apply gates to
            time: Time for the operations
            
        Returns:
            List of CX operations
        """
        operations = []
        
        for point in lattice_points:
            if self.ancilla_type_enum == AncillaType.X:
                ancilla_qubit = qubit_system.get_qubit_index(point, "X_anc")
            else:  # Z
                ancilla_qubit = qubit_system.get_qubit_index(point, "Z_anc")
            
            # Get ancilla position
            ancilla_pos = qubit_system.get_qubit_position(ancilla_qubit)
            position = ancilla_pos.coords if ancilla_pos else None
            
            # Get the connected data qubit using the centralized method
            data_qubit, data_label = self.get_connected_data_qubit(qubit_system, point)
            
            # Get individual positions for control/target
            data_pos = qubit_system.get_qubit_position(data_qubit)
            
            # Create CX gate with positions
            if self.ancilla_type_enum == AncillaType.X:
                operations.append(CX(time, ancilla_qubit, data_qubit,
                                   position=position,
                                   control_position=position,
                                   target_position=data_pos.coords if data_pos else None))
            else:  # Z
                operations.append(CX(time, data_qubit, ancilla_qubit,
                                   position=position,
                                   control_position=data_pos.coords if data_pos else None,
                                   target_position=position))
        
        return operations


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
