import unittest

import numpy as np

import symft


class MeasurementSamplingTest(unittest.TestCase):
    def test_deterministic_measurement_samples(self):
        circuit = symft.Circuit("X 0\nM 0\n")

        samples = circuit.sample(shots=8, seed=1)

        self.assertEqual(samples.shape, (8, 1))
        self.assertEqual(samples.dtype, np.bool_)
        self.assertTrue(np.all(samples))

    def test_zero_shots_preserves_record_width(self):
        circuit = symft.Circuit("M 0\n")

        samples = circuit.sample(shots=0)

        self.assertEqual(samples.shape, (0, 1))
        self.assertEqual(samples.dtype, np.bool_)

    def test_multiple_records(self):
        circuit = symft.Circuit("X 0\nM 0\nX 1\nM 1\n")

        samples = circuit.sample(shots=4, seed=2)

        self.assertEqual(samples.shape, (4, 2))
        self.assertTrue(np.all(samples))

    def test_bit_packed_output_spans_multiple_bytes(self):
        text = "\n".join(f"X {q}\nM {q}" for q in range(9))
        circuit = symft.Circuit(text)

        packed = circuit.sample(shots=2, seed=3, bit_packed=True)

        self.assertEqual(packed.shape, (2, 2))
        self.assertEqual(packed.dtype, np.uint8)
        self.assertTrue(np.all(packed[:, 0] == 0xFF))
        self.assertTrue(np.all((packed[:, 1] & 0x01) == 0x01))
        self.assertTrue(np.all((packed[:, 1] & 0xFE) == 0x00))

    def test_seed_reproducibility(self):
        circuit = symft.Circuit("H 0\nM 0\n")

        a = circuit.sample(shots=32, seed=123)
        b = circuit.sample(shots=32, seed=123)

        self.assertTrue(np.array_equal(a, b))

    def test_single_and_batch_deterministic_samples_match(self):
        circuit = symft.Circuit("X 0\nM 0\n")

        single = circuit.sample(shots=16, seed=5, batch=False)
        batch = circuit.sample(shots=16, seed=5, batch=True, batch_size=4)

        self.assertTrue(np.array_equal(single, batch))

    def test_module_level_sample_accepts_text(self):
        samples = symft.sample("X 0\nM 0\n", shots=3, seed=7)

        self.assertEqual(samples.shape, (3, 1))
        self.assertTrue(np.all(samples))

    def test_compiled_measurement_sampler_reuse(self):
        circuit = symft.Circuit("X 0\nM 0\n")
        sampler = circuit.compile_sampler()

        a = sampler.sample(shots=4, seed=11)
        b = sampler.sample(shots=4, seed=11)

        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(sampler.num_qubits, 1)
        self.assertEqual(sampler.num_measurements, 1)
        self.assertEqual(sampler.num_detectors, 0)
        self.assertGreaterEqual(sampler.max_active_qubits, 0)

    def test_compiled_batch_measurement_sampler(self):
        circuit = symft.Circuit("X 0\nM 0\n")
        sampler = circuit.compile_sampler(batch=True, batch_size=2)

        samples = sampler.sample(shots=5, seed=13)

        self.assertEqual(samples.shape, (5, 1))
        self.assertTrue(np.all(samples))


if __name__ == "__main__":
    unittest.main()
