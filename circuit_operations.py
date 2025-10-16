"""
Circuit operation classes for building syndrome extraction circuits with time ordering.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Tuple
from dataclasses import dataclass


@dataclass
class CircuitOperation(ABC):
    """
    Base class for all circuit operations with time ordering.
    """
    time: float
    
    @abstractmethod
    def to_stim(self) -> str:
        """Convert operation to Stim instruction string."""
        pass
    
    @abstractmethod
    def affected_qubits(self) -> List[int]:
        """Return list of qubit indices affected by this operation."""
        pass


@dataclass
class Reset(CircuitOperation):
    """
    Reset operation for ancilla qubits.
    """
    qubit: int
    basis: str  # 'X' or 'Z'
    
    def to_stim(self) -> str:
        """Convert to Stim reset instruction."""
        if self.basis == 'X':
            return f"RX {self.qubit}"
        elif self.basis == 'Z':
            return f"R {self.qubit}"
        else:
            raise ValueError(f"Invalid basis '{self.basis}'. Must be 'X' or 'Z'")
    
    def affected_qubits(self) -> List[int]:
        """Return list of affected qubits."""
        return [self.qubit]


@dataclass
class Measure(CircuitOperation):
    """
    Measurement operation for ancilla qubits.
    """
    qubit: int
    basis: str  # 'X' or 'Z'
    
    def to_stim(self) -> str:
        """Convert to Stim measurement instruction."""
        if self.basis == 'X':
            return f"MX {self.qubit}"
        elif self.basis == 'Z':
            return f"M {self.qubit}"
        else:
            raise ValueError(f"Invalid basis '{self.basis}'. Must be 'X' or 'Z'")
    
    def affected_qubits(self) -> List[int]:
        """Return list of affected qubits."""
        return [self.qubit]


@dataclass
class CX(CircuitOperation):
    """
    Controlled-X gate operation.
    """
    control: int
    target: int
    
    def to_stim(self) -> str:
        """Convert to Stim CX instruction."""
        return f"CX {self.control} {self.target}"
    
    def affected_qubits(self) -> List[int]:
        """Return list of affected qubits."""
        return [self.control, self.target]


@dataclass
class Depolarize2(CircuitOperation):
    """
    2-qubit depolarizing noise operation.
    """
    qubit1: int
    qubit2: int
    probability: float
    
    def to_stim(self) -> str:
        """Convert to Stim DEPOLARIZE2 instruction."""
        return f"DEPOLARIZE2({self.probability}) {self.qubit1} {self.qubit2}"
    
    def affected_qubits(self) -> List[int]:
        """Return list of affected qubits."""
        return [self.qubit1, self.qubit2]


@dataclass
class Detector(CircuitOperation):
    """
    Detector operation for syndrome extraction.
    """
    qubit: int
    previous_measurement_index: int  # Index of previous measurement (for rec[-k])
    current_measurement_index: int   # Index of current measurement (for rec[-k])
    
    def to_stim(self) -> str:
        """Convert to Stim DETECTOR instruction."""
        return f"DETECTOR rec[-{self.current_measurement_index}] rec[-{self.previous_measurement_index}]"
    
    def affected_qubits(self) -> List[int]:
        """Return list of affected qubits."""
        return [self.qubit]


@dataclass
class Observable(CircuitOperation):
    """
    Observable (logical operator) for Stim.
    """
    observable_id: int  # Numeric ID for the observable
    measurement_indices: List[int]  # Indices for rec[-k] references
    
    def to_stim(self) -> str:
        """Convert to Stim OBSERVABLE_INCLUDE instruction."""
        rec_refs = ' '.join([f"rec[-{idx}]" for idx in self.measurement_indices])
        return f"OBSERVABLE_INCLUDE({self.observable_id}) {rec_refs}" if rec_refs else f"OBSERVABLE_INCLUDE({self.observable_id})"
    
    def affected_qubits(self) -> List[int]:
        """Observables don't directly affect qubits."""
        return []


@dataclass
class Tick(CircuitOperation):
    """
    TICK instruction for time separation in Stim.
    """
    
    def to_stim(self) -> str:
        """Convert to Stim TICK instruction."""
        return "TICK"
    
    def affected_qubits(self) -> List[int]:
        """TICK affects no qubits."""
        return []
