import unittest

import numpy as np

import symft


class StimFeatureSmokeTest(unittest.TestCase):
    def test_measurement_record_feedback(self):
        circuit = symft.Circuit("M !0\nCX rec[-1] 1\nM 1\n")

        samples = circuit.sample(shots=3, seed=1)

        self.assertEqual(samples.shape, (3, 2))
        self.assertTrue(np.all(samples[:, 0]))
        self.assertTrue(np.all(samples[:, 1]))

    def test_t_gate_gives_nondeterministic_x_measurement(self):
        circuit = symft.Circuit("T 0\nMX 0\n")

        samples = circuit.sample(shots=200, seed=1)
        ones = int(samples[:, 0].sum())

        self.assertGreater(ones, 20)
        self.assertLess(ones, 180)

    def test_deterministic_noise(self):
        circuit = symft.Circuit("X_ERROR(1) 0\nM 0\n")

        samples = circuit.sample(shots=5, seed=1)

        self.assertTrue(np.all(samples))

    def test_mpad(self):
        circuit = symft.Circuit("MPAD 1 0\n")

        samples = circuit.sample(shots=2, seed=1)

        self.assertEqual(samples.shape, (2, 2))
        self.assertTrue(np.all(samples[:, 0]))
        self.assertFalse(np.any(samples[:, 1]))

    def test_rotation_and_extended_noise_smoke(self):
        circuit = symft.Circuit(
            "R_X(0.5) 0\n"
            "HERALDED_PAULI_CHANNEL_1(1, 0, 0, 0) 0\n"
            "M 0\n"
        )

        samples = circuit.sample(shots=4, seed=2)

        self.assertEqual(samples.shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
