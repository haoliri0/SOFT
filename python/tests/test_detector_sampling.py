import unittest

import numpy as np

import symft


class DetectorSamplingTest(unittest.TestCase):
    def test_quiet_detector_samples_false(self):
        circuit = symft.Circuit("M 0\nDETECTOR rec[-1]\n")

        detectors = circuit.sample_detectors(shots=5, seed=1)

        self.assertEqual(detectors.shape, (5, 1))
        self.assertEqual(detectors.dtype, np.bool_)
        self.assertFalse(np.any(detectors))

    def test_fired_detector_samples_true(self):
        circuit = symft.Circuit("X 0\nM 0\nDETECTOR rec[-1]\n")

        detectors = circuit.sample_detectors(shots=5, seed=1)

        self.assertTrue(np.all(detectors))

    def test_compiled_detector_sampler(self):
        circuit = symft.Circuit("X 0\nM 0\nDETECTOR rec[-1]\n")
        sampler = circuit.compile_sampler()

        detectors = sampler.sample_detectors(shots=4, seed=2)

        self.assertEqual(detectors.shape, (4, circuit.num_detectors))
        self.assertTrue(np.all(detectors))

    def test_bit_packed_detector_output_spans_multiple_bytes(self):
        text = "X 0\nM 0\n" + "\n".join("DETECTOR rec[-1]" for _ in range(9))
        circuit = symft.Circuit(text)

        packed = circuit.sample_detectors(shots=2, seed=3, bit_packed=True)

        self.assertEqual(packed.shape, (2, 2))
        self.assertEqual(packed.dtype, np.uint8)
        self.assertTrue(np.all(packed[:, 0] == 0xFF))
        self.assertTrue(np.all((packed[:, 1] & 0x01) == 0x01))


if __name__ == "__main__":
    unittest.main()
