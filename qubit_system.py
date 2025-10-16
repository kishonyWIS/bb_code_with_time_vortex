"""
Qubit system for managing qubits on lattice points.
Each lattice point hosts four qubits: L, R (data qubits) and X_anc, Z_anc (ancilla qubits).
"""

import numpy as np
from typing import Dict, Tuple, List, Union, Optional
from lattice import Lattice, Point


class QubitSystem:
    """
    Manages qubits on a lattice where each point hosts four qubits:
    - L, R: data qubits (Left and Right)
    - X_anc, Z_anc: ancilla qubits for X-type and Z-type stabilizer measurements
    """
    
    QUBIT_LABELS = ['L', 'R', 'X_anc', 'Z_anc']
    
    def __init__(self, lattice: Lattice):
        """
        Initialize qubit system for the given lattice.
        
        Args:
            lattice: The underlying lattice structure
        """
        self.lattice = lattice
        self._qubit_index_map: Dict[Tuple[Point, str], int] = {}
        self._next_index = 0
    
    def get_qubit_index(self, point: Union[Point, np.ndarray, List[Union[int, float]]], 
                       label: str) -> int:
        """
        Get the Stim qubit index for a qubit at the given point and label.
        
        Args:
            point: Lattice point
            label: Qubit label ('L', 'R', 'X_anc', 'Z_anc')
            
        Returns:
            Unique integer index for the qubit
        """
        if label not in self.QUBIT_LABELS:
            raise ValueError(f"Invalid qubit label '{label}'. Must be one of {self.QUBIT_LABELS}")
        
        # Normalize the point to canonical form
        normalized_point = self.lattice.normalize_point(point)
        key = (normalized_point, label)
        
        if key not in self._qubit_index_map:
            self._qubit_index_map[key] = self._next_index
            self._next_index += 1
        
        return self._qubit_index_map[key]
    
    def get_qubit_pair(self, point: Union[Point, np.ndarray, List[Union[int, float]]], 
                      label1: str, label2: str) -> Tuple[int, int]:
        """
        Get qubit indices for two qubits at the same point.
        
        Args:
            point: Lattice point
            label1, label2: Qubit labels
            
        Returns:
            Tuple of qubit indices
        """
        return (self.get_qubit_index(point, label1), 
                self.get_qubit_index(point, label2))
    
    def get_shifted_qubit_pair(self, point: Union[Point, np.ndarray, List[Union[int, float]]], 
                              label1: str, shift: Union[np.ndarray, List[Union[int, float]]], 
                              label2: str) -> Tuple[int, int]:
        """
        Get qubit indices for qubits at different points (one shifted).
        
        Args:
            point: Base lattice point
            label1: Label for qubit at base point
            shift: Shift vector
            label2: Label for qubit at shifted point
            
        Returns:
            Tuple of (qubit1_index, qubit2_index)
        """
        shifted_point = self.lattice.get_shifted_point(point, shift)
        return (self.get_qubit_index(point, label1), 
                self.get_qubit_index(shifted_point, label2))
    
    def get_axis_shifted_pair(self, point: Union[Point, np.ndarray, List[Union[int, float]]], 
                             label1: str, axis: int, label2: str) -> Tuple[int, int]:
        """
        Get qubit indices for qubits at points shifted along an axis.
        
        Args:
            point: Base lattice point
            label1: Label for qubit at base point
            axis: Axis to shift along (0-based)
            label2: Label for qubit at shifted point
            
        Returns:
            Tuple of (qubit1_index, qubit2_index)
        """
        shift = self.lattice.get_axis_vector(axis)
        return self.get_shifted_qubit_pair(point, label1, shift, label2)
    
    def get_all_qubits_at_point(self, point: Union[Point, np.ndarray, List[Union[int, float]]]) -> Dict[str, int]:
        """
        Get all qubit indices at a given point.
        
        Args:
            point: Lattice point
            
        Returns:
            Dictionary mapping qubit labels to indices
        """
        normalized_point = self.lattice.normalize_point(point)
        return {label: self.get_qubit_index(normalized_point, label) 
                for label in self.QUBIT_LABELS}
    
    def get_total_qubit_count(self) -> int:
        """Get the total number of qubits in the system."""
        return self._next_index
    
    def get_qubit_info(self, index: int) -> Optional[Tuple[Point, str]]:
        """
        Get the lattice point and label for a given qubit index.
        
        Args:
            index: Qubit index
            
        Returns:
            Tuple of (lattice_point, label) or None if not found
        """
        for (point, label), qubit_index in self._qubit_index_map.items():
            if qubit_index == index:
                return (point, label)
        return None
    
    def get_all_qubits(self) -> Dict[Tuple[Point, str], int]:
        """
        Get all qubit mappings.
        
        Returns:
            Dictionary mapping (point, label) to qubit index
        """
        return self._qubit_index_map.copy()
    
    def get_x_stabilizer_support(self, point: Union[Point, np.ndarray, List[Union[int, float]]]) -> List[Tuple[int, str]]:
        """
        Get the qubit indices and types for X stabilizer support at a given point.
        
        X stabilizer support includes:
        - On-site: both L and R qubits at the point
        - Even axes: R qubits at point + positive shift along each axis
        - Odd axes: L qubits at point + positive shift along each axis
        
        Args:
            point: Lattice point for the stabilizer
            
        Returns:
            List of (qubit_index, qubit_type) tuples
        """
        normalized_point = self.lattice.normalize_point(point)
        support = []
        
        # On-site qubits (both L and R)
        support.append((self.get_qubit_index(normalized_point, 'L'), 'L'))
        support.append((self.get_qubit_index(normalized_point, 'R'), 'R'))
        
        # Even axes: R qubits with positive shift
        for axis in range(0, self.lattice.dimension, 2):
            shift = self.lattice.get_axis_vector(axis)
            shifted_point = self.lattice.get_shifted_point(normalized_point, shift)
            support.append((self.get_qubit_index(shifted_point, 'R'), 'R'))
        
        # Odd axes: L qubits with positive shift
        for axis in range(1, self.lattice.dimension, 2):
            shift = self.lattice.get_axis_vector(axis)
            shifted_point = self.lattice.get_shifted_point(normalized_point, shift)
            support.append((self.get_qubit_index(shifted_point, 'L'), 'L'))
        
        return support
    
    def get_z_stabilizer_support(self, point: Union[Point, np.ndarray, List[Union[int, float]]]) -> List[Tuple[int, str]]:
        """
        Get the qubit indices and types for Z stabilizer support at a given point.
        
        Z stabilizer support includes:
        - On-site: both L and R qubits at the point
        - Even axes: L qubits at point + negative shift along each axis
        - Odd axes: R qubits at point + negative shift along each axis
        
        Args:
            point: Lattice point for the stabilizer
            
        Returns:
            List of (qubit_index, qubit_type) tuples
        """
        normalized_point = self.lattice.normalize_point(point)
        support = []
        
        # On-site qubits (both L and R)
        support.append((self.get_qubit_index(normalized_point, 'L'), 'L'))
        support.append((self.get_qubit_index(normalized_point, 'R'), 'R'))
        
        # Even axes: L qubits with negative shift
        for axis in range(0, self.lattice.dimension, 2):
            shift = -self.lattice.get_axis_vector(axis)
            shifted_point = self.lattice.get_shifted_point(normalized_point, shift)
            support.append((self.get_qubit_index(shifted_point, 'L'), 'L'))
        
        # Odd axes: R qubits with negative shift
        for axis in range(1, self.lattice.dimension, 2):
            shift = -self.lattice.get_axis_vector(axis)
            shifted_point = self.lattice.get_shifted_point(normalized_point, shift)
            support.append((self.get_qubit_index(shifted_point, 'R'), 'R'))
        
        return support

    def __repr__(self) -> str:
        """String representation."""
        return f"QubitSystem(lattice={self.lattice}, qubits={self._next_index})"
