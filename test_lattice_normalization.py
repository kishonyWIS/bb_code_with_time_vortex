"""
Comprehensive tests for lattice point normalization and enumeration.
"""

import numpy as np
from lattice import Lattice, Point


def test_orthogonal_lattice():
    """Test orthogonal lattice enumeration."""
    print("=== Test 1: Orthogonal Lattice ===")
    lattice = Lattice([[2, 0], [0, 3]])
    points = lattice.get_all_lattice_points()
    
    expected_count = 2 * 3  # det([[2,0],[0,3]]) = 6
    actual_count = len(points)
    
    print(f"Expected points: {expected_count}")
    print(f"Actual points: {actual_count}")
    print(f"Points: {[tuple(p.coords) for p in points]}")
    assert actual_count == expected_count, f"Expected {expected_count} points, got {actual_count}"
    print("✓ PASSED\n")


def test_rotated_lattice():
    """Test rotated lattice enumeration."""
    print("=== Test 2: Rotated Lattice ===")
    lattice = Lattice([[3, 3], [3, -3]])
    points = lattice.get_all_lattice_points()
    
    expected_count = 18  # |det([[3,3],[3,-3]])| = 18
    actual_count = len(points)
    
    print(f"Expected points: {expected_count}")
    print(f"Actual points: {actual_count}")
    assert actual_count == expected_count, f"Expected {expected_count} points, got {actual_count}"
    print("✓ PASSED\n")


def test_integer_point_normalization():
    """Test normalization of integer points."""
    print("=== Test 3: Integer Point Normalization ===")
    lattice = Lattice([[2, 0], [0, 2]])
    
    test_cases = [
        ([0, 0], [0, 0]),
        ([1, 1], [1, 1]),
        ([2, 0], [0, 0]),  # On boundary, should wrap
        ([0, 2], [0, 0]),  # On boundary, should wrap
        ([3, 3], [1, 1]),
        ([-1, -1], [1, 1]),
    ]
    
    for input_point, expected in test_cases:
        result = lattice.normalize_point(input_point)
        result_tuple = tuple(result.coords)
        print(f"  {input_point} -> {result_tuple} (expected {expected})")
        assert result_tuple == tuple(expected), f"Expected {expected}, got {result_tuple}"
    
    print("✓ PASSED\n")


def test_non_integer_point_normalization():
    """Test normalization of non-integer points."""
    print("=== Test 4: Non-integer Point Normalization ===")
    lattice = Lattice([[2, 0], [0, 2]])
    
    test_points = [
        [1.5, 0.5],
        [2.0, 0.0],
        [2.1, 0.1],
        [-0.5, 1.5],
        [0.3, 1.7],
    ]
    
    for point in test_points:
        normalized = lattice.normalize_point(point)
        # After rounding and normalization, should be in [0,2) x [0,2)
        in_domain = all(0 <= c < 2 for c in normalized.coords)
        print(f"  {point} -> {tuple(normalized.coords)}, in domain: {in_domain}")
        assert in_domain, f"Point {normalized.coords} not in fundamental domain [0,2) x [0,2)"
    
    print("✓ PASSED\n")


def test_normalization_consistency():
    """Test that normalizing twice gives the same result."""
    print("=== Test 5: Normalization Consistency ===")
    lattice = Lattice([[3, 3], [3, -3]])
    
    test_points = [
        [0, 0],
        [5, 5],
        [-2, 3],
        [1.5, 2.7],
    ]
    
    for point in test_points:
        first = lattice.normalize_point(point)
        second = lattice.normalize_point(first)
        
        print(f"  {point} -> {tuple(first.coords)} -> {tuple(second.coords)}")
        assert np.array_equal(first.coords, second.coords), \
            f"Normalization not idempotent: {first.coords} != {second.coords}"
    
    print("✓ PASSED\n")


def test_boundary_cases():
    """Test points exactly on boundaries."""
    print("=== Test 6: Boundary Cases ===")
    lattice = Lattice([[4, 0], [0, 4]])
    
    # Points exactly on the upper boundary should wrap to 0
    boundary_cases = [
        ([4, 0], [0, 0]),
        ([0, 4], [0, 0]),
        ([4, 4], [0, 0]),
        ([3.999, 0], [3.999, 0]),  # Very close to boundary, but still in [0,4)
        ([0.001, 0], [0.001, 0]),  # Very close to 0, stays as is
    ]
    
    for input_point, expected in boundary_cases:
        result = lattice.normalize_point(input_point)
        result_tuple = tuple(result.coords)
        print(f"  {input_point} -> {result_tuple} (expected {expected})")
        assert result_tuple == tuple(expected), f"Expected {expected}, got {result_tuple}"
    
    print("✓ PASSED\n")


def test_different_dimensions():
    """Test lattices in different dimensions."""
    print("=== Test 7: Different Dimensions ===")
    
    # 1D lattice
    lattice_1d = Lattice([[5]])
    points_1d = lattice_1d.get_all_lattice_points()
    print(f"  1D lattice [5]: {len(points_1d)} points (expected 5)")
    assert len(points_1d) == 5
    
    # 3D lattice
    lattice_3d = Lattice([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    points_3d = lattice_3d.get_all_lattice_points()
    print(f"  3D lattice [[2,0,0],[0,2,0],[0,0,2]]: {len(points_3d)} points (expected 8)")
    assert len(points_3d) == 8
    
    print("✓ PASSED\n")


def test_non_orthogonal_3d():
    """Test a non-orthogonal 3D lattice."""
    print("=== Test 8: Non-orthogonal 3D Lattice ===")
    lattice = Lattice([[2, 1, 0], [0, 2, 1], [1, 0, 2]])
    points = lattice.get_all_lattice_points()
    
    det = abs(int(np.round(np.linalg.det(lattice.lattice_vectors))))
    print(f"  Determinant: {det}")
    print(f"  Points found: {len(points)}")
    assert len(points) == det, f"Expected {det} points, got {len(points)}"
    print("✓ PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("LATTICE NORMALIZATION AND ENUMERATION TESTS")
    print("=" * 60)
    print()
    
    try:
        test_orthogonal_lattice()
        test_rotated_lattice()
        test_integer_point_normalization()
        test_non_integer_point_normalization()
        test_normalization_consistency()
        test_boundary_cases()
        test_different_dimensions()
        test_non_orthogonal_3d()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓✓✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise

