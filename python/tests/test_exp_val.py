import numpy as np

import symft


def test_exp_val_is_non_destructive_and_returns_values():
    with_probe = symft.Circuit("\n".join(("H 0", "EXP_VAL X0", "M 0")))
    without_probe = symft.Circuit("\n".join(("H 0", "M 0")))
    records, values = with_probe.sample_with_expectations(32, seed=17)
    reference = without_probe.sample(32, seed=17)
    assert np.array_equal(records, reference)
    assert np.allclose(values, 1.0)


def test_exp_val_pauli_products_and_dormant_support():
    circuit = symft.Circuit("\n".join(("H 0", "CX 0 1", "EXP_VAL Z0*Z1 X0*X1 X2")))
    _, values = circuit.sample_with_expectations(1)
    assert np.allclose(values[0], (1.0, 1.0, 0.0))
