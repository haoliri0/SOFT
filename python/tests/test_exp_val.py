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


def test_exp_val_uses_dense_state_when_component_planning_would_apply():
    circuit = symft.Circuit(
        "\n".join(
            [*(f"H {q}" for q in range(13)), *(f"T {q}" for q in range(13))]
            + ["EXP_VAL " + "*".join(f"X{q}" for q in range(13))]
        )
    )
    _, values = circuit.sample_with_expectations(1)
    assert np.allclose(values, 2 ** (-13 / 2))


def test_exp_val_compiled_batch_sampler():
    circuit = symft.Circuit(
        "\n".join(
            (
                "EXP_VAL Z0 X0",
                "H 0",
                "T 0",
                "EXP_VAL X0 Y0 Z0",
                "T_DAG 0",
                "H 0",
                "M 0",
            )
        )
    )
    sampler = circuit.compile_sampler(batch=True, batch_size=3)

    records, values = sampler.sample_with_expectations(7, seed=23)

    assert "batch=True" in repr(sampler)
    assert not np.any(records)
    assert np.allclose(values[:, :2], (1.0, 0.0))
    assert np.allclose(values[:, 2:], (2**-0.5, 2**-0.5, 0.0))
