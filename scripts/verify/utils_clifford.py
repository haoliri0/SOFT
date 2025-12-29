from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit_aer import AerSimulator


def compute_statevector(
    clifford: Clifford,
    entries: Iterable[tuple[tuple[bool, ...], complex]] | None = None,
) -> np.ndarray:
    circuit = clifford.to_circuit()
    circuit.save_statevector()
    simulator = AerSimulator(method='statevector')
    result = simulator.run(circuit, shots=1).result()

    state = result.data()['statevector']
    if entries is None:
        return state

    entries_state = []
    entries = tuple(entries)
    for ddd, _ in entries:
        circuit = QuantumCircuit(clifford.num_qubits)
        circuit.initialize(state)
        for i, d in enumerate(ddd):
            if not d:
                continue
            for q in range(clifford.num_qubits):
                x = clifford.destab_x[i, q]
                z = clifford.destab_z[i, q]
                if (x, z) == (0, 0):
                    continue
                if (x, z) == (0, 1):
                    circuit.z(q)
                if (x, z) == (1, 0):
                    circuit.x(q)
                if (x, z) == (1, 1):
                    circuit.y(q)
            if clifford.phase[i]:
                circuit.global_phase += np.pi
        circuit.save_statevector()
        result = simulator.run(circuit, shots=1).result()
        entry_state = result.data()['statevector']
        entries_state.append(entry_state)

    if len(entries_state) == 0:
        return None

    entries_state = tuple(
        np.asarray(amp * entry_state)
        for entry_state, (_, amp) in zip(entries_state, entries))
    return np.sum(entries_state, axis=0)


def compute_statevector_from_soft(
    table: tuple[str, ...],
    entries: Iterable[tuple[tuple[bool, ...], complex]],
) -> np.ndarray:
    clifford = Clifford(tuple(row[0:1] + row[1:][::-1] for row in table))
    return compute_statevector(clifford, entries)
