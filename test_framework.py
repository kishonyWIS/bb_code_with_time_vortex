"""
Simple test script to verify the lattice qubit framework functionality.
"""

from lattice import Lattice, Point
from qubit_system import QubitSystem


def test_basic_functionality():
    """Test basic lattice and qubit functionality."""
    print("Testing basic functionality...")
    
    # Test 1: Basic lattice creation and point operations
    lattice = Lattice([[3, 0], [0, 3]])
    p1 = Point([1, 1])
    p2 = Point([4, 1])  # Should be equivalent to p1
    
    assert lattice.are_equivalent(p1, p2), "Points should be equivalent"
    assert lattice.normalize_point(p2) == p1, "Normalization should work"
    
    # Test 2: Qubit system
    qubit_system = QubitSystem(lattice)
    
    # Same point should give same qubit indices
    idx1 = qubit_system.get_qubit_index(p1, 'L')
    idx2 = qubit_system.get_qubit_index(p2, 'L')
    assert idx1 == idx2, "Equivalent points should have same qubit indices"
    
    # Test 3: Shifted qubit pairs
    q1, q2 = qubit_system.get_axis_shifted_pair(p1, 'X_anc', 0, 'L')
    assert isinstance(q1, int) and isinstance(q2, int), "Should return integer indices"
    
    print("✓ All basic tests passed!")


def test_4d_lattice():
    """Test 4D lattice functionality."""
    print("Testing 4D lattice...")
    
    # Create 4D lattice
    lattice_vectors = [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]]
    lattice = Lattice(lattice_vectors)
    qubit_system = QubitSystem(lattice)
    
    # Test point in 4D
    point = Point([1, 1, 1, 1])
    shifted_point = lattice.get_shifted_point(point, [1, 0, 0, 0])
    
    # Test all qubit types
    qubits = qubit_system.get_all_qubits_at_point(point)
    assert len(qubits) == 4, "Should have 4 qubits per point"
    assert all(label in qubits for label in ['L', 'R', 'X_anc', 'Z_anc']), "Should have all qubit types"
    
    # Test shifted pairs for all axes
    for axis in range(4):
        q1, q2 = qubit_system.get_axis_shifted_pair(point, 'X_anc', axis, 'L')
        assert isinstance(q1, int) and isinstance(q2, int), f"Axis {axis} should return integer indices"
    
    print("✓ 4D lattice tests passed!")


if __name__ == "__main__":
    test_basic_functionality()
    test_4d_lattice()
    print("\n🎉 All tests passed! The framework is working correctly.")
