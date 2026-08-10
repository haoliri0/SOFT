import math
import unittest
from concurrent.futures import ThreadPoolExecutor

import symft


class CountsSamplingTest(unittest.TestCase):
    def test_summary_keys_and_no_discard(self):
        circuit = symft.Circuit("M 0\n")

        result = circuit.sample_counts(shots=3, batch=False)

        for key in [
            "shots",
            "discarded",
            "accepted",
            "logical_errors",
            "discard_rate",
            "logical_error_rate",
            "active_threads",
            "timing",
        ]:
            self.assertIn(key, result)
        self.assertEqual(result["shots"], 3)
        self.assertEqual(result["accepted"], 3)
        self.assertEqual(result["discarded"], 0)
        self.assertEqual(result["logical_errors"], 0)

    def test_logical_error_count(self):
        circuit = symft.Circuit("X 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n")

        result = circuit.sample_counts(shots=5, batch=False)

        self.assertEqual(result["accepted"], 5)
        self.assertEqual(result["logical_errors"], 5)
        self.assertEqual(result["logical_error_rate"], 1.0)

    def test_detector_discard_count(self):
        circuit = symft.Circuit("X 0\nM 0\nDETECTOR rec[-1]\n")

        result = circuit.sample_counts(shots=5, batch=False)

        self.assertEqual(result["discarded"], 5)
        self.assertEqual(result["accepted"], 0)
        self.assertTrue(math.isnan(result["logical_error_rate"]))

    def test_batch_counts(self):
        circuit = symft.Circuit("M 0\nDETECTOR rec[-1]\n")

        result = circuit.sample_counts(shots=9, batch=True, batch_size=2)

        self.assertEqual(result["shots"], 9)
        self.assertEqual(result["accepted"], 9)
        self.assertEqual(result["discarded"], 0)

    def test_batch_postselection_discards_fired_detector(self):
        circuit = symft.Circuit("X 0\nM 0\nDETECTOR rec[-1]\n")

        result = circuit.sample_counts(
            shots=8,
            batch=True,
            batch_size=4,
            postselect_detectors=True,
        )

        self.assertEqual(result["discarded"], 8)
        self.assertEqual(result["accepted"], 0)

    def test_reference_sample_normalizes_detector_postselection(self):
        circuit = symft.Circuit("X 0\nM 0\nDETECTOR rec[-1]\n")

        for batch in [False, True]:
            with self.subTest(batch=batch):
                result = circuit.sample_counts(
                    shots=8,
                    batch=batch,
                    batch_size=4,
                    postselect_detectors=True,
                    reference_sample=True,
                )

                self.assertEqual(result["discarded"], 0)
                self.assertEqual(result["accepted"], 8)

    def test_reference_sample_normalizes_logical_observable(self):
        circuit = symft.Circuit("X 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n")

        raw = circuit.sample_counts(shots=5, batch=False)
        normalized = circuit.sample_counts(
            shots=5,
            batch=False,
            reference_sample=True,
        )

        self.assertEqual(raw["logical_errors"], 5)
        self.assertEqual(normalized["logical_errors"], 0)

    def test_reference_sample_reports_all_detector_and_observable_bits(self):
        circuit = symft.Circuit(
            "X 0\n"
            "M 0\n"
            "DETECTOR rec[-1]\n"
            "OBSERVABLE_INCLUDE(0)\n"
            "OBSERVABLE_INCLUDE(1) rec[-1]\n"
        )

        reference = circuit.reference_sample()

        self.assertEqual(reference["detectors"], (True,))
        self.assertEqual(reference["observables"], (False, True))

    def test_full_expected_observable_vector_uses_selected_index(self):
        circuit = symft.Circuit(
            "X 0\nM 0\nOBSERVABLE_INCLUDE(0)\nOBSERVABLE_INCLUDE(1) rec[-1]\n"
        )

        raw = circuit.sample_counts(shots=4, observable=1, batch=True)
        normalized = circuit.sample_counts(
            shots=4,
            observable=1,
            batch=True,
            expected_observables=[False, True],
        )

        self.assertEqual(raw["logical_errors"], 4)
        self.assertEqual(normalized["logical_errors"], 0)

    def test_reference_sample_uses_selected_observable_index(self):
        circuit = symft.Circuit(
            "X 0\nM 0\nOBSERVABLE_INCLUDE(0)\nOBSERVABLE_INCLUDE(1) rec[-1]\n"
        )

        for batch in [False, True]:
            with self.subTest(batch=batch):
                raw = circuit.sample_counts(shots=4, observable=1, batch=batch)
                normalized = circuit.sample_counts(
                    shots=4,
                    observable=1,
                    batch=batch,
                    reference_sample=True,
                )

                self.assertEqual(raw["logical_errors"], 4)
                self.assertEqual(normalized["logical_errors"], 0)

    def test_explicit_expected_vectors_normalize_counts(self):
        circuit = symft.Circuit(
            "X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
        )

        result = circuit.sample_counts(
            shots=6,
            batch=True,
            batch_size=3,
            postselect_detectors=True,
            expected_detectors=[True],
            expected_observables=[True],
        )

        self.assertEqual(result["discarded"], 0)
        self.assertEqual(result["accepted"], 6)
        self.assertEqual(result["logical_errors"], 0)

    def test_prepared_counts_sampler_reuse_and_info(self):
        circuit = symft.Circuit("M 0\n")
        sampler = circuit.compile_counts_sampler(
            batch=True,
            batch_size=2,
            sample_chunk_shots=4,
            threads=1,
        )

        a = sampler.sample(shots=3, stream_id=0)
        b = sampler.sample(shots=4, stream_id=1)

        self.assertEqual(a["shots"], 3)
        self.assertEqual(b["shots"], 4)
        self.assertEqual(sampler.info["num_measurements"], 1)
        self.assertEqual(sampler.info["batch_size"], 2)
        self.assertEqual(sampler.info["sample_chunk_shots"], 4)
        self.assertEqual(sampler.info["threads"], 1)
        self.assertEqual(sampler.info["backend"], "batch")
        self.assertFalse(sampler.info["reference_normalized"])
        self.assertIn("sample_s", sampler.preprocessing_timing)

    def test_prepared_counts_sampler_reports_reference_mode(self):
        circuit = symft.Circuit("X 0\nM 0\nDETECTOR rec[-1]\n")
        sampler = circuit.compile_counts_sampler(reference_sample=True)

        self.assertTrue(sampler.info["reference_normalized"])

    def test_stream_id_reproducibility(self):
        circuit = symft.Circuit("H 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n")
        sampler = circuit.compile_counts_sampler(batch=False)

        a = sampler.sample(shots=128, stream_id=9)
        b = sampler.sample(shots=128, stream_id=9)

        self.assertEqual(a["logical_errors"], b["logical_errors"])
        self.assertEqual(a["discarded"], b["discarded"])

    def test_default_counts_parameters(self):
        circuit = symft.Circuit("M 0\n")

        result = circuit.sample_counts()
        sampler = circuit.compile_counts_sampler(batch=True)

        self.assertEqual(result["shots"], 1)
        self.assertGreater(sampler.info["batch_size"], 0)
        self.assertGreaterEqual(sampler.info["sample_chunk_shots"], sampler.info["batch_size"])

    def test_compiled_counts_sampler_serializes_concurrent_calls(self):
        circuit = symft.Circuit(
            "H 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"
        )
        sampler = circuit.compile_counts_sampler(batch=True, batch_size=64)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(sampler.sample, 4096, 17) for _ in range(2)]
            results = [future.result() for future in futures]

        self.assertEqual(results[0]["discarded"], results[1]["discarded"])
        self.assertEqual(results[0]["logical_errors"], results[1]["logical_errors"])

    def test_invalid_cuda_mode_is_rejected(self):
        circuit = symft.Circuit("M 0\n")

        with self.assertRaisesRegex(ValueError, "cuda_mode"):
            circuit.compile_counts_sampler(cuda_mode="unknown")



if __name__ == "__main__":
    unittest.main()
