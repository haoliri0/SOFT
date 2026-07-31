import symft


circuit = symft.Circuit(
    """
H 0
M 0
"""
)

print(circuit)
print(circuit.sample(shots=8, seed=5))
print(circuit.compile_sampler().sample(shots=8, seed=5, bit_packed=True))
print(circuit.sample_counts(shots=1000))
