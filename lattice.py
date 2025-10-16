"""
Core lattice point and lattice classes for bivariate bicycle quantum code simulation.
"""

import numpy as np
from typing import List, Tuple, Union
from dataclasses import dataclass
from itertools import product
from fractions import Fraction
from math import floor


@dataclass(frozen=True)
class Point:
    """
    Represents a point in D-dimensional space with coordinates stored as a NumPy array.
    Immutable and hashable for use in dictionaries and sets.
    Coordinates can be integer or float.
    """
    coords: np.ndarray
    
    def __init__(self, coords):
        # Ensure coords is a 1D array, preserve original dtype (int or float)
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)  # Don't force dtype=int
        elif coords.ndim != 1:
            raise ValueError("Coordinates must be 1D")
        # Don't force dtype conversion - preserve int/float
        
        # Use object.__setattr__ to bypass frozen dataclass restriction
        object.__setattr__(self, 'coords', coords)
    
    def __add__(self, other: Union['Point', np.ndarray, List[Union[int, float]]]) -> 'Point':
        """Add another point or vector to this point."""
        if isinstance(other, Point):
            return Point(self.coords + other.coords)
        else:
            other_coords = np.array(other)  # Don't force dtype=int
            if len(other_coords) != len(self.coords):
                raise ValueError("Dimension mismatch")
            return Point(self.coords + other_coords)
    
    def __sub__(self, other: Union['Point', np.ndarray, List[Union[int, float]]]) -> 'Point':
        """Subtract another point or vector from this point."""
        if isinstance(other, Point):
            return Point(self.coords - other.coords)
        else:
            other_coords = np.array(other)  # Don't force dtype=int
            if len(other_coords) != len(self.coords):
                raise ValueError("Dimension mismatch")
            return Point(self.coords - other_coords)
    
    def __mul__(self, scalar: Union[int, float]) -> 'Point':
        """Multiply point coordinates by a scalar."""
        return Point(self.coords * scalar)
    
    def __rmul__(self, scalar: Union[int, float]) -> 'Point':
        """Right multiplication by scalar."""
        return self.__mul__(scalar)
    
    def __eq__(self, other) -> bool:
        """Equality comparison (element-wise)."""
        if not isinstance(other, Point):
            return False
        return np.array_equal(self.coords, other.coords)
    
    def __hash__(self) -> int:
        """Hash function for use in dictionaries and sets."""
        return hash(tuple(self.coords))
    
    def normalized_hash(self, lattice: 'Lattice') -> int:
        """Hash function using normalized coordinates for periodic boundary conditions."""
        normalized = lattice.normalize_point(self)
        return hash(tuple(normalized.coords))
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Point({self.coords.tolist()})"
    
    def __len__(self) -> int:
        """Return the dimension of the point."""
        return len(self.coords)
    
    def __getitem__(self, index: int) -> int:
        """Access coordinate by index."""
        return self.coords[index]
    
    def __iter__(self):
        """Iterate over coordinates."""
        return iter(self.coords)


class Lattice:
    """
    Represents a D-dimensional lattice with periodic boundary conditions.
    Points that differ by integer multiples of lattice vectors are considered equivalent.
    """
    
    def __init__(self, lattice_vectors: List[List[int]]):
        """
        Initialize lattice with given lattice vectors.
        
        Args:
            lattice_vectors: List of D linearly independent integer vectors
        """
        self.lattice_vectors = np.array(lattice_vectors, dtype=int)
        self.dimension = self.lattice_vectors.shape[1]
        
        if len(self.lattice_vectors) != self.dimension:
            raise ValueError(f"Need {self.dimension} lattice vectors, got {len(self.lattice_vectors)}")
        
        # Check linear independence
        if np.linalg.matrix_rank(self.lattice_vectors) < self.dimension:
            raise ValueError("Lattice vectors must be linearly independent")
        
        # Compute the lattice matrix and its inverse for normalization
        self.lattice_matrix = self.lattice_vectors.T
        try:
            self.lattice_matrix_inv = np.linalg.inv(self.lattice_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Lattice vectors must be linearly independent")
    
    def _invert_matrix_fraction(self, M):
        """Helper method to invert a matrix using Fraction-based Gaussian elimination."""
        m = [[Fraction(int(v)) for v in row] for row in M]
        n = len(m)
        I = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
        for col in range(n):
            piv = col
            while piv < n and m[piv][col] == 0:
                piv += 1
            if piv == n:
                raise ValueError("Lattice vectors must be linearly independent")
            if piv != col:
                m[col], m[piv] = m[piv], m[col]
                I[col], I[piv] = I[piv], I[col]
            pv = m[col][col]
            invpv = Fraction(1, 1) / pv
            for j in range(n):
                m[col][j] *= invpv
                I[col][j] *= invpv
            for r in range(n):
                if r == col:
                    continue
                f = m[r][col]
                if f == 0:
                    continue
                for j in range(n):
                    m[r][j] -= f * m[col][j]
                    I[r][j] -= f * I[col][j]
        return I
    
    def _matvec(self, M, v):
        """Helper method for matrix-vector multiplication."""
        n = len(M)
        return [sum(M[i][j] * v[j] for j in range(n)) for i in range(n)]
    
    def _matvec_int(self, M, v):
        """Helper method for matrix-vector multiplication with integer result."""
        n = len(M)
        return [sum(M[i][j] * v[j] for j in range(n)) for i in range(n)]
    
    def _convert_to_appropriate_type(self, values, original_is_int, epsilon):
        """Helper method to convert values to appropriate types (preserving int vs float)."""
        result = []
        for val in values:
            if hasattr(val, 'is_Integer') and val.is_Integer:
                result.append(int(val))
            elif hasattr(val, 'is_Rational') and val.is_Rational:
                # Try to convert to int if it's close enough
                float_val = float(val)
                int_val = int(round(float_val))
                if abs(float_val - int_val) < epsilon:
                    result.append(int_val)
                else:
                    result.append(float_val)
            else:
                result.append(float(val))
        
        # If original input was all integers, try to keep result as integers
        if original_is_int:
            # Check if all results are close to integers
            all_close_to_int = all(abs(r - round(r)) < epsilon for r in result)
            if all_close_to_int:
                result = [int(round(r)) for r in result]
        
        return result
    
    def normalize_point(self, point: Union[Point, np.ndarray, List[Union[int, float]], List[float]], epsilon: float = 1e-12) -> Point:
        """
        Reduce a point (integer, rational, or float) modulo the lattice into the 
        half-open fundamental domain P(A) = {A t | t in [0,1)^n}.
        
        Uses exact arithmetic (sympy or fractions.Fraction) to avoid numerical errors.
        
        Args:
            point: Point to normalize (can be integer or non-integer coordinates)
            epsilon: Small tolerance for boundary cases (values near 1.0 are treated as inside [0,1))
            
        Returns:
            Canonical representative in the fundamental domain as a Point
        """
        if isinstance(point, Point):
            x = point.coords
        else:
            x = np.array(point)
        
        if len(x) != self.dimension:
            raise ValueError(f"Point dimension {len(x)} doesn't match lattice dimension {self.dimension}")
        
        n = self.dimension
        A = self.lattice_vectors.tolist()  # Convert to list for sympy/Fraction
        
        # Try SymPy (exact and robust)
        try:
            import sympy as sp
            A_sym = sp.Matrix(A)
            if A_sym.det() == 0:
                raise ValueError("Lattice vectors must be linearly independent")
            
            x_sym = sp.Matrix(list(x))
            y = A_sym.LUsolve(x_sym)  # A^{-1} x
            
            # Floor with epsilon so values like 0.9999999999999999 are treated as inside [0,1)
            t = sp.Matrix([int(sp.floor(y[i] + epsilon)) for i in range(n)])
            p = x_sym - A_sym * t
            
            # Convert to appropriate type (preserve int vs float)
            original_is_int = all(isinstance(x[i], (int, np.integer)) for i in range(len(x)))
            result = self._convert_to_appropriate_type(p, original_is_int, epsilon)
            
            # After rounding to integers, normalize again to ensure we're in [0, basis_element)
            # This handles cases where rounding puts us on the boundary
            result_point = Point(result)
            
            # Check if we need to normalize again (recursively, but with integer input now)
            # We check if any coordinate is >= the corresponding lattice vector diagonal
            # For a general lattice, we need to check if the point is actually in the fundamental domain
            # by checking if 0 <= A^-1 * result < 1 componentwise
            y_check = A_sym.LUsolve(sp.Matrix(result))
            needs_renormalization = any(float(y_check[i]) < -epsilon or float(y_check[i]) >= 1 - epsilon for i in range(n))
            
            if needs_renormalization:
                # Recursively normalize (but this time with integer input)
                return self.normalize_point(result_point)
            
            return result_point
        
        except Exception:
            # Fallback: Fraction-based exact path
            def to_fraction(v):
                if isinstance(v, Fraction):
                    return v
                if isinstance(v, (int, np.integer)):
                    return Fraction(int(v), 1)
                return Fraction(v).limit_denominator()
            
            xf = [to_fraction(v) for v in x]
            
            Ainv = self._invert_matrix_fraction(A)
            y = self._matvec(Ainv, xf)
            
            # epsilon-aware floor for Fractions
            def floor_eps(val, eps=epsilon):
                f = float(val)
                return floor(f + eps)
            
            t = [int(floor_eps(y[i])) for i in range(n)]
            p = [xf[i] - sum(Fraction(int(A[i][j])) * t[j] for j in range(n)) for i in range(n)]
            
            # Convert to appropriate type (preserve int vs float)
            original_is_int = all(isinstance(x[i], (int, np.integer)) for i in range(len(x)))
            result = self._convert_to_appropriate_type(p, original_is_int, epsilon)
            
            # After rounding to integers, normalize again to ensure we're in the fundamental domain
            result_point = Point(result)
            
            # Check if we need to normalize again
            y_check = self._matvec(Ainv, result)
            needs_renormalization = any(float(y_check[i]) < -epsilon or float(y_check[i]) >= 1 - epsilon for i in range(n))
            
            if needs_renormalization:
                # Recursively normalize (but this time with integer input)
                return self.normalize_point(result_point)
            
            return result_point
    
    def are_equivalent(self, p1: Union[Point, np.ndarray, List[Union[int, float]]], 
                      p2: Union[Point, np.ndarray, List[Union[int, float]]]) -> bool:
        """
        Check if two points are equivalent under periodic boundary conditions.
        
        Args:
            p1, p2: Points to compare
            
        Returns:
            True if points are equivalent
        """
        return self.normalize_point(p1) == self.normalize_point(p2)
    
    def get_shifted_point(self, point: Union[Point, np.ndarray, List[Union[int, float]]], 
                         shift: Union[np.ndarray, List[Union[int, float]]]) -> Point:
        """
        Get a point shifted by the given vector, normalized to canonical form.
        
        Args:
            point: Base point
            shift: Shift vector
            
        Returns:
            Normalized shifted point
        """
        if isinstance(point, Point):
            shifted_coords = point.coords + np.array(shift)  # Don't force dtype=int
        else:
            shifted_coords = np.array(point) + np.array(shift)  # Don't force dtype=int
        
        return self.normalize_point(shifted_coords)
    
    def get_axis_vector(self, axis: int) -> np.ndarray:
        """
        Get the unit vector along the specified axis.
        
        Args:
            axis: Axis index (0-based)
            
        Returns:
            Unit vector along the axis
        """
        if axis < 0 or axis >= self.dimension:
            raise ValueError(f"Axis {axis} out of range [0, {self.dimension-1}]")
        
        vec = np.zeros(self.dimension, dtype=int)
        vec[axis] = 1
        return vec
    
    def get_all_lattice_points(self) -> List[Point]:
        """
        Return all integer points in the half-open fundamental domain P(A) = {A t | t in [0,1)^n} ∩ Z^n.
        
        Uses Smith Normal Form (via sympy) for exact and efficient enumeration.
        Falls back to a brute-force method when sympy is unavailable.
        
        Returns:
            List of all unique lattice points in the fundamental domain (exactly |det(A)| points)
        """
        n = self.dimension
        basis = self.lattice_vectors.tolist()
        
        # Helper: compute determinant using Fraction-based Gaussian elimination
        def det_fraction(M):
            m = [[Fraction(x) for x in row] for row in M]
            n = len(m)
            det = Fraction(1, 1)
            for i in range(n):
                p = i
                while p < n and m[p][i] == 0:
                    p += 1
                if p == n:
                    return Fraction(0, 1)
                if p != i:
                    m[i], m[p] = m[p], m[i]
                    det *= -1
                piv = m[i][i]
                det *= piv
                for r in range(i + 1, n):
                    if m[r][i] == 0:
                        continue
                    f = m[r][i] / piv
                    for c in range(i, n):
                        m[r][c] -= f * m[i][c]
            return det
        
        D = det_fraction(basis)
        if D == 0:
            raise ValueError("Lattice vectors must be linearly independent (nonzero determinant).")
        detA = abs(int(D))
        
        if detA > 1000:
            raise ValueError(f"Lattice has too many points ({detA}) for enumeration. "
                           "Consider using a smaller lattice.")
        
        # Try the fast exact method (sympy SNF)
        try:
            import sympy as sp
            A_sym = sp.Matrix(basis)
            # smith_normal_form returns (D, U, V) with U*A*V = D
            D_sym, U_sym, V_sym = A_sym.smith_normal_form()
            # Invert U and A over the rationals (exact)
            Uinv = U_sym.inv()  # unimodular -> integer inverse
            Ainv = A_sym.inv()  # rational exact
            
            # diagonal invariants
            diags = [int(D_sym[i, i]) for i in range(n)]
            # Defensive: if any diagonal is 0 (shouldn't happen for full-rank), raise
            if any(d <= 0 for d in diags):
                raise ValueError("Smith Normal Form produced a zero diagonal; basis may be rank-deficient.")
            
            points = []
            seen = set()
            for coords in product(*[range(d) for d in diags]):
                r = sp.Matrix(coords)
                q = Uinv * r                     # an integer coset representative in Z^n
                y = Ainv * q                     # real vector
                t = sp.Matrix([int(sp.floor(y[i])) for i in range(n)])
                p = q - A_sym * t                # in the fundamental domain
                tup = tuple(int(p[i]) for i in range(n))
                if tup not in seen:
                    seen.add(tup)
                    points.append(Point(list(tup)))
            
            points.sort(key=lambda pt: tuple(pt.coords))
            # sanity: should be exactly |det A|
            if len(points) != detA:
                # Extremely rare; but dedup + SNF path should yield |det A|
                # Fall back to brute-force if mismatch
                raise RuntimeError("SNF path produced unexpected count; falling back.")
            
            return points
        
        except Exception:
            # Fallback (no sympy): enumerate x in [0,detA-1]^n and fold into P(A).
            # Implement exact inverse via Fractions (Gauss-Jordan).
            Ainv_frac = self._invert_matrix_fraction(basis)
            
            points_set = set()
            # Helper to floor Fractions componentwise
            def floor_vec(v):
                return [int(floor(v[i])) for i in range(n)]
            
            for x in product(range(detA), repeat=n):
                # y = A^{-1} x
                y = self._matvec(Ainv_frac, list(x))
                t = floor_vec(y)
                # p = x - A t
                At = self._matvec_int(basis, t)
                p = tuple(x[i] - At[i] for i in range(n))
                points_set.add(p)
                if len(points_set) == detA:
                    # Done (we've collected all unique points)
                    pass
            
            points = [Point(list(p)) for p in sorted(points_set)]
            return points
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Lattice(dimension={self.dimension}, vectors={self.lattice_vectors.tolist()})"
