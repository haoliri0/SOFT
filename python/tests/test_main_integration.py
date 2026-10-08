"""Regression coverage for porting the opt-in backends onto main's APIs."""
import inspect
import numpy as np
import pytest

import symft


def counts(result):
    return tuple(result[key] for key in ("shots", "discarded", "accepted", "logical_errors"))


def test_public_signatures_keep_main_keywords_and_scope():
    for method in (symft.Circuit.sample_counts, symft.Circuit.compile_counts_sampler):
        parameters = inspect.signature(method).parameters
        assert parameters["reference_sample"].default is False
        assert parameters["cpu_backend"].default == "legacy"
        assert parameters["cpu_real_gauge"].default is True
    assert "cpu_backend" not in inspect.signature(symft.Circuit.compile_sampler).parameters


@pytest.mark.parametrize("real", [False, True])
@pytest.mark.parametrize("hoist", [False, True])
@pytest.mark.parametrize("shots", [0, 1, 63, 64, 65, 257])
def test_compiled_cpu_reference_normalization(real, hoist, shots):
    circuit = symft.Circuit("X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(1) rec[-1]\n")
    options = dict(cpu_backend="compiled", postselect_detectors=True, threads=1,
                   observable=1, cpu_real_gauge=real, cpu_hoist_detectors=hoist,
                   sample_chunk_shots=65)
    for reference in (dict(reference_sample=True),
                      dict(expected_detectors=[True], expected_observables=[False, True])):
        sampler = circuit.compile_counts_sampler(**options, **reference)
        assert sampler.info["cpu_compiled"]
        assert sampler.info["reference_normalized"]
        assert counts(sampler.sample(shots, stream_id=17)) == (shots, 0, shots, 0)
        assert counts(circuit.sample_counts(shots, seed=17, **options, **reference)) == (shots, 0, shots, 0)
    raw = circuit.sample_counts(shots, seed=17, **options)
    assert counts(raw) == (shots, shots, 0, 0)


@pytest.mark.parametrize("hoist", [False, True])
def test_compiled_reference_multiword_and_contradictory_rows(hoist):
    # The 65th detector and observable 65 must use the second reference word.
    circuit = symft.Circuit("X 0\nM 0\n" + "DETECTOR rec[-1]\n" * 65 +
                            "OBSERVABLE_INCLUDE(65) rec[-1]\n")
    sampler = circuit.compile_counts_sampler(cpu_backend="compiled", postselect_detectors=True,
                                             observable=65, reference_sample=True,
                                             cpu_hoist_detectors=hoist)
    assert sampler.info["cpu_compiled"]
    assert counts(sampler.sample(65, stream_id=3)) == (65, 0, 65, 0)
    opposite = circuit.compile_counts_sampler(cpu_backend="compiled", postselect_detectors=True,
                                              observable=65, expected_detectors=[True] * 64 + [False],
                                              cpu_hoist_detectors=hoist)
    assert counts(opposite.sample(65, stream_id=3)) == (65, 65, 0, 0)


def test_compiled_reference_preserves_noise_and_nonzero_logical_counts():
    circuit = symft.Circuit("X 0\nX_ERROR(0.2) 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
    options = dict(cpu_backend="compiled", postselect_detectors=True, sample_chunk_shots=65)
    raw = circuit.sample_counts(1001, seed=99, **options)
    normalized = circuit.sample_counts(1001, seed=99, reference_sample=True, **options)
    assert raw["logical_errors"] + normalized["logical_errors"] == 1001
    assert 100 < normalized["logical_errors"] < 300


def test_compiled_counts_with_expectation_probes_falls_back_without_collapsing():
    circuit = symft.Circuit("H 0\nT 0\nEXP_VAL X0 Y0 Z0\nT_DAG 0\nH 0\nM 0\n"
                            "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
    sampler = circuit.compile_counts_sampler(cpu_backend="compiled", postselect_detectors=True)
    assert not sampler.info["cpu_compiled"]
    assert sampler.info["cpu_fallback_reason"] == "expectation_values_present"
    assert counts(sampler.sample(257, stream_id=4)) == (257, 0, 257, 0)


@pytest.mark.skipif(not symft.cuda_enabled(), reason="requires a CUDA Python build and GPU")
@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize("with_probes", [False, True])
@pytest.mark.parametrize("mode", ["gpu_presample_expressions", "gpu", "cpu_presampled"])
def test_cuda_jit_options_preserve_records_and_expectations(monkeypatch, symbolic, with_probes, mode):
    for name in ("SYMFT_CUDA_JIT", "SYMFT_CUDA_PACKED_NOISE", "SYMFT_CUDA_SPARSE_NOISE",
                 "SYMFT_CUDA_SCALAR_NOISE"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("SYMFT_JIT_SYMBOLIC", "1" if symbolic else "0")
    monkeypatch.setenv("SYMFT_JIT_ROW_REDUCE", "1" if symbolic else "0")
    text = "X 1\nH 0\nT 0\n"
    if with_probes:
        text += "EXP_VAL X0 Y0 Z0 X2 Z1\n"
    text += "T_DAG 0\nH 0\nM 0 1\n"
    circuit = symft.Circuit(text)
    options = dict(cuda=True, cuda_mode=mode, shots_per_launch=65)
    sampler = circuit.compile_sampler(**options)
    for take_sample in (lambda: sampler.sample_with_expectations(257, seed=19),
                        lambda: circuit.sample_with_expectations(257, seed=19, **options)):
        records, values = take_sample()
        assert records.shape == (257, 2)
        assert np.array_equal(records, np.tile([False, True], (257, 1)))
        if with_probes:
            assert np.allclose(values, [2**-0.5, 2**-0.5, 0.0, 0.0, -1.0], atol=1e-12)
        else:
            assert values.shape == (257, 0)
    # Counts with probes must also retain non-destructive semantics.
    assert counts(circuit.sample_counts(65, seed=19, **options)) == (65, 0, 65, 0)


@pytest.mark.skipif(not symft.cuda_enabled(), reason="requires a CUDA Python build and GPU")
@pytest.mark.parametrize("shots", [0, 1, 7, 31, 32, 63, 64, 65, 257])
def test_cuda_jit_aggregate_boundaries(monkeypatch, shots):
    for name in ("SYMFT_CUDA_JIT", "SYMFT_CUDA_PACKED_NOISE", "SYMFT_CUDA_SPARSE_NOISE",
                 "SYMFT_CUDA_SCALAR_NOISE", "SYMFT_JIT_SYMBOLIC", "SYMFT_JIT_ROW_REDUCE"):
        monkeypatch.setenv(name, "1")
    options = dict(cuda=True, cuda_mode="gpu_presample_expressions", shots_per_launch=65)
    for postselect in (False, True):
        logical = symft.Circuit("X 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
        assert counts(logical.sample_counts(shots, postselect_detectors=postselect, **options)) == (shots, 0, shots, shots)
        rejected = symft.Circuit("X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
        assert counts(rejected.sample_counts(shots, postselect_detectors=postselect, **options)) == (shots, shots, 0, 0)
