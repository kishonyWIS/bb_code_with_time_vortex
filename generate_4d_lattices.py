"""
Generate all unique 4D lattices with bounded determinant.

Enumeration strategy:
1. Enumerate all m, l such that m*l <= max_det
2. First vector: (0, m, 0, 0)
3. Second vector: (l, q, 0, 0) for each 0 <= q < m
4. Enumerate all (a, b) and (c, d) lying within the parallelogram spanned by (0, m) and (l, q)
5. Third vector: (a, b, -1, 0)
6. Fourth vector: (c, d, 0, -1)

Constraint: |det| <= max_det (default max_det=3, since n = 4 * |det| and n <= 30)
"""

import numpy as np
import itertools
from tqdm import tqdm
import sys
import os

# Add torus_embedding to path to import lll_reduce
torus_embedding_path = os.path.join(os.path.dirname(__file__), 'torus_embedding')
if os.path.exists(torus_embedding_path):
    sys.path.insert(0, torus_embedding_path)
    from lll_reduce import lll_reduce
else:
    # Fallback: try importing from current directory structure
    try:
        from torus_embedding.lll_reduce import lll_reduce
    except ImportError:
        raise ImportError("Could not import lll_reduce. Make sure torus_embedding/lll_reduce.py exists.")


def normalize_signs(B):
    """
    Normalize signs of basis vectors for canonical form.
    For each vector, ensure first non-zero element is positive.
    """
    B = B.copy()
    for i in range(B.shape[0]):
        # Find first non-zero element
        for j in range(B.shape[1]):
            if B[i, j] != 0:
                if B[i, j] < 0:
                    B[i] *= -1
                break
    return B


def matrix_to_canonical_tuple(B):
    """
    Convert a matrix to a canonical tuple for hashing.
    First normalize signs, then convert to tuple.
    """
    B_normalized = normalize_signs(B)
    # Sort rows lexicographically for canonical form
    # Convert to tuple of tuples for hashing
    rows = [tuple(row) for row in B_normalized]
    rows_sorted = sorted(rows)
    return tuple(rows_sorted)


def get_points_in_parallelogram(m, l, q):
    """
    Get all integer points (a, b) in the half-open parallelogram
    spanned by (0, m) and (l, q).
    
    The parallelogram is: {s*(0,m) + t*(l,q) : s, t in [0, 1)}
    Area = m*l (absolute value of determinant)
    
    The transformation from (s,t) to (a,b) is:
    [a]   [0  l] [s]
    [b] = [m q] [t]
    
    So: a = t*l, b = s*m + t*q
    
    The inverse transformation (from (a,b) to (s,t)) is:
    [s] = (1/det) * [[q, -l], [-m, 0]] * [a]  where det = -m*l
    [t]                                    [b]
    
    So: s = (l*b - q*a) / (m*l)
        t = a/l
    
    For (s, t) in [0, 1) x [0, 1):
    - 0 <= a/l < 1  =>  0 <= a < l  (if l > 0)
    - 0 <= (l*b - q*a) / (m*l) < 1  =>  0 <= l*b - q*a < m*l
                                         =>  q*a <= l*b < q*a + m*l
    
    Args:
        m, l, q: Parameters defining the parallelogram (m, l > 0)
    
    Returns:
        List of (a, b) tuples representing integer points in the parallelogram
    """
    points = []
    
    if l == 0:
        # Degenerate case
        return [(0, 0)]
    
    # Bounding box for the parallelogram
    corners = [(0, 0), (l, q), (0, m), (l, q + m)]
    min_a = min(c[0] for c in corners)
    max_a = max(c[0] for c in corners)
    min_b = min(c[1] for c in corners)
    max_b = max(c[1] for c in corners)
    
    det_2d = -m * l  # Determinant of [[0, l], [m, q]]
    
    for a in range(min_a, max_a + 1):
        # Check t = a/l in [0, 1)
        if a < 0 or a >= l:
            continue
        
        for b in range(min_b, max_b + 1):
            # Check s = (l*b - q*a) / (m*l) in [0, 1)
            numerator = l * b - q * a
            if numerator < 0 or numerator >= m * l:
                continue
            
            points.append((a, b))
    
    return points


def generate_4d_lattices(
    max_det=3,
    min_det=1,
    output_file='unique_4d_lattices.csv'
):
    """
    Generate all unique 4D lattices with the specified enumeration strategy.
    
    Enumeration:
    1. Enumerate all m, l such that m*l <= max_det
    2. First vector: (0, m, 0, 0)
    3. Second vector: (l, q, 0, 0) for each 0 <= q < m
    4. Enumerate all (a, b) and (c, d) in parallelogram spanned by (0, m) and (l, q)
    5. Third vector: (a, b, -1, 0)
    6. Fourth vector: (c, d, 0, -1)
    
    Args:
        max_det: Maximum absolute determinant (default: 3)
        min_det: Minimum absolute determinant (default: 1)
        output_file: Output CSV file path
    
    Returns:
        List of unique lattices (each as tuple of (abs_det, list of 4 vectors)), sorted by determinant
    """
    seen_reduced = set()
    unique_lattices = []
    
    # Statistics
    total_candidates = 0
    rank_failures = 0
    det_failures = 0
    duplicates = 0
    det_distribution = {}  # Track determinant distribution
    
    print(f"Generating 4D lattices with determinant range: [{min_det}, {max_det}]")
    
    # Enumerate all m, l such that m*l <= max_det
    ml_pairs = []
    for m in range(1, max_det + 1):
        for l in range(1, max_det + 1):
            if m * l <= max_det:
                ml_pairs.append((m, l))
    
    print(f"Total (m, l) pairs with m*l <= {max_det}: {len(ml_pairs)}")
    
    total_combinations = 0
    for m, l in ml_pairs:
        for q in range(m):  # 0 <= q < m
            points = get_points_in_parallelogram(m, l, q)
            total_combinations += len(points) * len(points)
    
    print(f"Total matrix combinations to test: {total_combinations}")
    
    for m, l in tqdm(ml_pairs, desc="m, l pairs"):
        # First vector: (0, m, 0, 0)
        v1 = np.array([0, m, 0, 0], dtype=int)
        
        for q in range(m):  # 0 <= q < m
            # Second vector: (l, q, 0, 0)
            v2 = np.array([l, q, 0, 0], dtype=int)
            
            # Get all points in the parallelogram
            points = get_points_in_parallelogram(m, l, q)
            
            # Enumerate all (a, b) and (c, d) in the parallelogram
            for a, b in points:
                # Third vector: (a, b, -1, 0)
                v3 = np.array([a, b, -1, 0], dtype=int)
                
                for c, d in points:
                    # Fourth vector: (c, d, 0, -1)
                    v4 = np.array([c, d, 0, -1], dtype=int)
                    
                    total_candidates += 1
                    
                    # Build matrix (rows are lattice vectors)
                    B = np.array([v1, v2, v3, v4], dtype=int)
                    
                    # Check linear independence (rank == 4)
                    if np.linalg.matrix_rank(B) < 4:
                        rank_failures += 1
                        continue
                    
                    # Check determinant constraint
                    det = np.linalg.det(B)
                    abs_det = abs(det)
                    if abs_det < min_det or abs_det > max_det or abs_det < 1e-10:  # Also skip near-zero det
                        det_failures += 1
                        continue
                    
                    # Perform LLL reduction
                    try:
                        B_reduced = lll_reduce(B, delta=0.75)
                        # Ensure integer type
                        B_reduced = np.round(B_reduced).astype(int)
                    except Exception as e:
                        # Skip if LLL reduction fails
                        continue
                    
                    # Convert to canonical form for hashing
                    canonical = matrix_to_canonical_tuple(B_reduced)
                    
                    # Check if we've seen this reduced form before
                    if canonical in seen_reduced:
                        duplicates += 1
                        continue
                    
                    # New unique lattice found
                    seen_reduced.add(canonical)
                    # Save original (unreduced) form with determinant for sorting
                    unique_lattices.append((abs_det, B.tolist()))
                    
                    # Track determinant distribution
                    det_int = int(round(abs(det)))
                    det_distribution[det_int] = det_distribution.get(det_int, 0) + 1
    
    # Print statistics
    print(f"\n=== Generation Statistics ===")
    print(f"Total candidates tested: {total_candidates}")
    print(f"Rank failures (not full rank): {rank_failures}")
    print(f"Determinant failures (|det| < {min_det} or |det| > {max_det} or det ≈ 0): {det_failures}")
    print(f"Duplicates (seen reduced form): {duplicates}")
    print(f"Unique lattices found: {len(unique_lattices)}")
    
    if det_distribution:
        print(f"\nDeterminant distribution:")
        for det_val in sorted(det_distribution.keys()):
            print(f"  |det| = {det_val}: {det_distribution[det_val]} lattices")
    
    # Sort by determinant before saving
    unique_lattices.sort(key=lambda x: x[0])  # Sort by abs_det (first element of tuple)
    
    # Save to file
    if unique_lattices:
        import csv
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['lattice_vec_0', 'lattice_vec_1', 'lattice_vec_2', 'lattice_vec_3'])
            for abs_det, lattice in unique_lattices:
                writer.writerow([str(lattice[0]), str(lattice[1]), str(lattice[2]), str(lattice[3])])
        print(f"\nSaved {len(unique_lattices)} unique lattices to {output_file} (sorted by increasing determinant)")
    else:
        print("\nNo unique lattices found!")
    
    return unique_lattices


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate unique 4D lattices')
    parser.add_argument('--max-det', type=int, default=10,
                        help='Maximum absolute determinant (also used as coordinate bound) [default: 5]')
    parser.add_argument('--min-det', type=int, default=6,
                        help='Minimum absolute determinant [default: 1]')
    parser.add_argument('--output', type=str, default='unique_4d_lattices.csv',
                        help='Output CSV file [default: unique_4d_lattices.csv]')
    
    args = parser.parse_args()
    
    generate_4d_lattices(
        max_det=args.max_det,
        min_det=args.min_det,
        output_file=args.output
    )

