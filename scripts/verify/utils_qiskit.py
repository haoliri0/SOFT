from collections.abc import Iterator
from itertools import count
from typing import Iterable

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from scripts.verify.utils_ops import Op


def make_qiskit_op(
    qc: QuantumCircuit,
    op: Op,
    states_i: Iterator[int],
    results_i: Iterator[int],
):
    match op:
        case ('CX', (control, target)):
            qc.cx(control, target)
        case ('X', target):
            qc.x(target)
        case ('Y', target):
            qc.y(target)
        case ('Z', target):
            qc.z(target)
        case ('H', target):
            qc.h(target)
        case ('S', target):
            qc.s(target)
        case ('SDG', target):
            qc.sdg(target)
        case ('T', target):
            qc.t(target)
        case ('TDG', target):
            qc.tdg(target)
        case ('MEASURE', target):
            result_i = next(results_i)
            qc.save_probabilities([target], label=f'prob_{result_i}')
            qc.measure(target, result_i)
        case _:
            raise ValueError(f"Unsupported operation: {op}")

    state_i = next(states_i)
    qc.save_statevector(label=f'state_{state_i}')


def make_qiskit_circuit(
    ops: Iterable[Op],
    qubits_n: int,
    results_n: int,
) -> QuantumCircuit:
    qc = QuantumCircuit(qubits_n, results_n)
    states_i = iter(count())
    results_i = iter(count())
    for op in ops:
        make_qiskit_op(qc, op, states_i, results_i)
    return qc


def run_qiskit_circuit(qc: QuantumCircuit):
    simulator = AerSimulator(method='statevector')
    simulator_result = simulator.run(qc, shots=1, memory=True).result()
    data = simulator_result.data()

    states = []
    for i in count():
        state_key = f'state_{i}'
        if state_key not in data:
            break
        states.append(data[state_key])
    states = tuple(states)

    if 'memory' in data:
        memory = simulator_result.get_memory()[0]
        results = reversed(memory)
        results = map(int, results)
        results = tuple(results)
    else:
        results = ()

    probs = tuple(data[f'prob_{i}'][r] for i, r in enumerate(results))

    return states, results, probs
