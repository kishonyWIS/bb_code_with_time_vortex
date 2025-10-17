"""
Logical operator computation for quantum error correction codes.
Uses binary field algebra to find logical operators from stabilizer generators.
"""

import numpy as np
import galois
from typing import List, Tuple, Dict
from lattice import Point, Lattice
from qubit_system import QubitSystem


class LogicalOperators:
    """
    Computes logical operators for a quantum error correction code
    from its stabilizer generators.
    """
    
    def __init__(self, qubit_system: QubitSystem, lattice_points: List[Point]):
        """
        Initialize logical operator computer.
        
        Args:
            qubit_system: Qubit system for getting stabilizer supports
            lattice_points: List of lattice points in the code
        """
        self.qubit_system = qubit_system
        self.lattice_points = lattice_points
        
        # Create ordered list of data qubits (L then R for each point)
        self.data_qubits = []
        self.data_qubit_to_index = {}
        
        for point in lattice_points:
            l_qubit = qubit_system.get_qubit_index(point, 'L')
            r_qubit = qubit_system.get_qubit_index(point, 'R')
            
            self.data_qubit_to_index[l_qubit] = len(self.data_qubits)
            self.data_qubits.append((point, 'L', l_qubit))
            
            self.data_qubit_to_index[r_qubit] = len(self.data_qubits)
            self.data_qubits.append((point, 'R', r_qubit))
        
        self.num_data_qubits = len(self.data_qubits)
        
        # Create mapping from qubit index to whether it's a data qubit
        self.is_data_qubit = {}
        for point in lattice_points:
            for label in ['L', 'R', 'X_anc', 'Z_anc']:
                qubit_idx = qubit_system.get_qubit_index(point, label)
                self.is_data_qubit[qubit_idx] = (label in ['L', 'R'])
        
        # Build parity check matrices
        self.hx, self.hz = self._build_parity_check_matrices()
        
        # Compute logical operators
        self.logical_x_ops, self.logical_z_ops = self._compute_logical_operators()
        print('logical_x_ops: ', self.logical_x_ops)
        print('logical_z_ops: ', self.logical_z_ops)
    
    def _build_parity_check_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build parity check matrices hx and hz for X and Z stabilizers.
        
        Each row corresponds to a stabilizer (one per lattice point for X and Z).
        Each column corresponds to a data qubit.
        
        Returns:
            Tuple of (hx, hz) matrices as numpy arrays with dtype=int
        """
        num_stabilizers = len(self.lattice_points)
        
        # Initialize matrices
        hx = np.zeros((num_stabilizers, self.num_data_qubits), dtype=int)
        hz = np.zeros((num_stabilizers, self.num_data_qubits), dtype=int)
        
        for stab_idx, point in enumerate(self.lattice_points):
            # Get X stabilizer support
            x_support = self.qubit_system.get_x_stabilizer_support(point)
            for qubit_idx, qubit_type in x_support:
                # Only include data qubits (L and R), not ancillas
                if self.is_data_qubit.get(qubit_idx, False):
                    col_idx = self.data_qubit_to_index[qubit_idx]
                    hx[stab_idx, col_idx] = 1
            
            # Get Z stabilizer support
            z_support = self.qubit_system.get_z_stabilizer_support(point)
            for qubit_idx, qubit_type in z_support:
                # Only include data qubits (L and R), not ancillas
                if self.is_data_qubit.get(qubit_idx, False):
                    col_idx = self.data_qubit_to_index[qubit_idx]
                    hz[stab_idx, col_idx] = 1
        
        # raise error if not
        if np.any((hx @ hz.T)%2 != 0):
            raise ValueError("hx and hz do not commute")

        return hx, hz
    
    def _compute_logical_operators(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute logical X and Z operators using binary field algebra.
        
        Logical Z operators are vectors orthogonal to hx but not in span of hz.
        Logical X operators are vectors orthogonal to hz but not in span of hx.
        
        Returns:
            Tuple of (logical_x_operators, logical_z_operators) as lists of binary vectors
        """
        GF = galois.GF(2)
        
        # Convert to galois arrays
        hx_gf = GF(self.hx)
        hz_gf = GF(self.hz)
        
        # Find logical Z operators
        # These are in the nullspace of hx but not in the row space of hz
        logical_z_ops = self._find_logical_operators(hx_gf, hz_gf)
        assert np.all((logical_z_ops @ self.hx.T)%2 == 0)
        
        # Find logical X operators
        # These are in the nullspace of hz but not in the row space of hx
        logical_x_ops = self._find_logical_operators(hz_gf, hx_gf)
        assert np.all((logical_x_ops @ self.hz.T)%2 == 0)

        # Convert back to numpy arrays
        logical_x_ops = [np.array(op, dtype=int) for op in logical_x_ops]
        logical_z_ops = [np.array(op, dtype=int) for op in logical_z_ops]

        assert len(logical_x_ops) == len(logical_z_ops)
        
        return logical_x_ops, logical_z_ops
    
    def _find_logical_operators(self, h_ortho: galois.FieldArray, 
                                h_span: galois.FieldArray) -> List[galois.FieldArray]:
        """
        Find logical operators in nullspace of h_ortho that are not in rowspace of h_span.
        
        Args:
            h_ortho: Matrix that logical operators must be orthogonal to
            h_span: Matrix whose rowspace logical operators must not be in
            
        Returns:
            List of logical operators (minimal independent set)
        """
        GF = galois.GF(2)
        
        # Find nullspace of h_ortho (vectors orthogonal to rows of h_ortho)
        nullspace_basis = h_ortho.null_space()
        
        if nullspace_basis.size == 0:
            return []
        
        # Find rowspace of h_span (stabilizer space)
        rowspace_basis = h_span.row_space()
        
        if rowspace_basis.size == 0:
            # If rowspace is empty, nullspace basis vectors are the logical operators
            return [v for v in nullspace_basis if not np.all(v == 0)]
        
        # Calculate expected dimension of quotient space
        expected_dim = nullspace_basis.shape[0] - rowspace_basis.shape[0]
        
        if expected_dim <= 0:
            return []
        
        # Use Gaussian elimination to find representatives for the quotient space
        # Combine rowspace and nullspace vectors
        all_vectors = GF(np.vstack([rowspace_basis, nullspace_basis]))
        
        # Perform Gaussian elimination to find a basis
        # The vectors that are not in the rowspace span are our logical operators
        basis_vectors = []
        current_matrix = GF(np.zeros((0, all_vectors.shape[1]), dtype=int))
        
        for i, vec in enumerate(all_vectors):
            if np.all(vec == 0):
                continue
                
            # Try to add this vector to the current basis
            extended_matrix = GF(np.vstack([current_matrix, vec]))
            
            if np.linalg.matrix_rank(extended_matrix) > np.linalg.matrix_rank(current_matrix):
                # This vector is independent
                basis_vectors.append(vec)
                current_matrix = extended_matrix
                
                # Stop when we have enough logical operators
                if len(basis_vectors) >= rowspace_basis.shape[0] + expected_dim:
                    break
        
        # The logical operators are the vectors beyond the rowspace dimension
        logical_ops = basis_vectors[rowspace_basis.shape[0]:]

        # test that the dimensions make sense
        assert len(logical_ops) == expected_dim
        
        return logical_ops
    
    def get_num_logical_qubits(self) -> int:
        """Get the number of logical qubits encoded by this code."""
        return len(self.logical_x_ops)
    
    def get_data_qubit_info(self, index: int) -> Tuple[Point, str, int]:
        """
        Get information about a data qubit by its index.
        
        Args:
            index: Index in the data qubit ordering
            
        Returns:
            Tuple of (lattice_point, label, qubit_index)
        """
        return self.data_qubits[index]
    
    def get_observable_qubits(self, logical_op: np.ndarray) -> List[int]:
        """
        Get the list of qubit indices where the logical operator has support.
        
        Args:
            logical_op: Binary vector representing the logical operator
            
        Returns:
            List of qubit indices
        """
        qubits = []
        for i, val in enumerate(logical_op):
            if val == 1:
                _, _, qubit_idx = self.data_qubits[i]
                qubits.append(qubit_idx)
        return qubits
    
    def print_summary(self):
        """Print a summary of the logical operators."""
        print(f"Code parameters:")
        print(f"  Number of data qubits: {self.num_data_qubits}")
        print(f"  Number of X stabilizers: {self.hx.shape[0]}")
        print(f"  Number of Z stabilizers: {self.hz.shape[0]}")
        print(f"  Number of logical X operators found: {len(self.logical_x_ops)}")
        print(f"  Number of logical Z operators found: {len(self.logical_z_ops)}")
        print(f"  Number of logical qubits: {self.get_num_logical_qubits()}")
        
        print(f"\nLogical X operators:")
        for i, op in enumerate(self.logical_x_ops):
            weight = np.sum(op)
            qubits = self.get_observable_qubits(op)
            print(f"  XL{i}: weight={weight}, qubits={qubits[:5]}{'...' if len(qubits) > 5 else ''}")
        
        print(f"\nLogical Z operators:")
        for i, op in enumerate(self.logical_z_ops):
            weight = np.sum(op)
            qubits = self.get_observable_qubits(op)
            print(f"  ZL{i}: weight={weight}, qubits={qubits[:5]}{'...' if len(qubits) > 5 else ''}")

