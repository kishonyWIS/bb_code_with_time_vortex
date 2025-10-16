"""
Test to verify that hashing and qubit indexing work correctly with periodic boundary conditions.
"""

from lattice import Lattice, LatticePoint
from qubit_system import QubitSystem


def test_periodic_hashing_and_indexing():
    """Test that equivalent points get the same qubit indices."""
    print("Testing periodic boundary conditions with hashing and qubit indexing...")
    
    # Create a 2D lattice with 4x4 periodicity
    lattice = Lattice([[4, 0], [0, 4]])
    qubit_system = QubitSystem(lattice)
    
    # Test points that should be equivalent due to periodicity
    p1 = LatticePoint([1, 1])
    p2 = LatticePoint([5, 1])   # Same as p1 (5-4=1)
    p3 = LatticePoint([1, 5])    # Same as p1 (5-4=1)
    p4 = LatticePoint([5, 5])    # Same as p1 (both coordinates wrap)
    p5 = LatticePoint([-3, 1])  # Same as p1 (-3+4=1)
    
    print(f"Point 1: {p1}")
    print(f"Point 2: {p2} (should be equivalent to p1)")
    print(f"Point 3: {p3} (should be equivalent to p1)")
    print(f"Point 4: {p4} (should be equivalent to p1)")
    print(f"Point 5: {p5} (should be equivalent to p1)")
    
    # Test equivalence
    assert lattice.are_equivalent(p1, p2), "p1 and p2 should be equivalent"
    assert lattice.are_equivalent(p1, p3), "p1 and p3 should be equivalent"
    assert lattice.are_equivalent(p1, p4), "p1 and p4 should be equivalent"
    assert lattice.are_equivalent(p1, p5), "p1 and p5 should be equivalent"
    
    # Test normalization
    normalized_p1 = lattice.normalize_point(p1)
    normalized_p2 = lattice.normalize_point(p2)
    normalized_p3 = lattice.normalize_point(p3)
    normalized_p4 = lattice.normalize_point(p4)
    normalized_p5 = lattice.normalize_point(p5)
    
    print(f"\nNormalized points:")
    print(f"p1 normalized: {normalized_p1}")
    print(f"p2 normalized: {normalized_p2}")
    print(f"p3 normalized: {normalized_p3}")
    print(f"p4 normalized: {normalized_p4}")
    print(f"p5 normalized: {normalized_p5}")
    
    # All normalized points should be identical
    assert normalized_p1 == normalized_p2 == normalized_p3 == normalized_p4 == normalized_p5, \
        "All normalized points should be identical"
    
    # Test qubit indexing - equivalent points should get the same qubit indices
    print(f"\nQubit indices:")
    
    # Get qubit indices for L qubits at all equivalent points
    idx1_L = qubit_system.get_qubit_index(p1, 'L')
    idx2_L = qubit_system.get_qubit_index(p2, 'L')
    idx3_L = qubit_system.get_qubit_index(p3, 'L')
    idx4_L = qubit_system.get_qubit_index(p4, 'L')
    idx5_L = qubit_system.get_qubit_index(p5, 'L')
    
    print(f"L qubit at p1: {idx1_L}")
    print(f"L qubit at p2: {idx2_L}")
    print(f"L qubit at p3: {idx3_L}")
    print(f"L qubit at p4: {idx4_L}")
    print(f"L qubit at p5: {idx5_L}")
    
    # All should be the same
    assert idx1_L == idx2_L == idx3_L == idx4_L == idx5_L, \
        "Equivalent points should get the same qubit indices"
    
    # Test all qubit types
    for qubit_type in ['L', 'R', 'X_anc', 'Z_anc']:
        idx1 = qubit_system.get_qubit_index(p1, qubit_type)
        idx2 = qubit_system.get_qubit_index(p2, qubit_type)
        assert idx1 == idx2, f"{qubit_type} qubits should have same index for equivalent points"
    
    print(f"✓ All equivalent points get the same qubit indices!")
    
    # Test that different points get different indices
    p_different = LatticePoint([2, 1])  # Different from p1
    idx_different = qubit_system.get_qubit_index(p_different, 'L')
    assert idx_different != idx1_L, "Different points should get different qubit indices"
    
    print(f"✓ Different points get different qubit indices!")
    
    # Test hashing - raw LatticePoint objects have different hashes
    point_set = {p1, p2, p3, p4, p5}
    print(f"\nSet of raw equivalent points has size: {len(point_set)}")
    print(f"Note: Raw LatticePoint objects have different hashes even if equivalent")
    
    # Test normalized hashing
    normalized_set = {lattice.normalize_point(p) for p in [p1, p2, p3, p4, p5]}
    print(f"Set of normalized equivalent points has size: {len(normalized_set)}")
    assert len(normalized_set) == 1, "Normalized equivalent points should hash to the same value"
    
    print(f"✓ Normalized hashing works correctly with periodic boundary conditions!")
    print(f"✓ QubitSystem automatically normalizes points, so indexing works correctly!")


def test_4d_periodic_indexing():
    """Test qubit indexing in 4D with periodic boundary conditions."""
    print(f"\nTesting 4D periodic indexing...")
    
    # Create 4D lattice
    lattice = Lattice([[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]])
    qubit_system = QubitSystem(lattice)
    
    # Test equivalent 4D points
    p1 = LatticePoint([1, 1, 1, 1])
    p2 = LatticePoint([4, 1, 1, 1])  # Equivalent to p1
    p3 = LatticePoint([1, 4, 1, 1])  # Equivalent to p1
    p4 = LatticePoint([1, 1, 4, 1])  # Equivalent to p1
    p5 = LatticePoint([1, 1, 1, 4])  # Equivalent to p1
    
    # Test qubit indexing
    idx1 = qubit_system.get_qubit_index(p1, 'X_anc')
    idx2 = qubit_system.get_qubit_index(p2, 'X_anc')
    idx3 = qubit_system.get_qubit_index(p3, 'X_anc')
    idx4 = qubit_system.get_qubit_index(p4, 'X_anc')
    idx5 = qubit_system.get_qubit_index(p5, 'X_anc')
    
    assert idx1 == idx2 == idx3 == idx4 == idx5, "4D equivalent points should get same qubit indices"
    
    print(f"✓ 4D periodic indexing works correctly!")


if __name__ == "__main__":
    test_periodic_hashing_and_indexing()
    test_4d_periodic_indexing()
    print(f"\n🎉 All periodic boundary condition tests passed!")
