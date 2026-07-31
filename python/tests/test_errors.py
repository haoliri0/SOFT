import unittest

import symft


class ErrorHandlingTest(unittest.TestCase):
    def test_invalid_stim_raises_symft_error(self):
        with self.assertRaises(symft.SymFTError):
            symft.Circuit("NOT_A_GATE 0\n")

    def test_sweep_controlled_operation_is_rejected(self):
        with self.assertRaisesRegex(symft.SymFTError, "sweep-controlled operations are not supported"):
            symft.Circuit("CX sweep[0] 0\n")

    def test_circuit_rejects_both_text_and_path(self):
        with self.assertRaises(TypeError):
            symft.Circuit("M 0\n", path="x.stim")

    def test_negative_shots_rejected(self):
        circuit = symft.Circuit("M 0\n")

        with self.assertRaises(ValueError):
            circuit.sample(shots=-1)
        with self.assertRaises(ValueError):
            circuit.sample_counts(shots=-1)
        with self.assertRaises(ValueError):
            circuit.sample_detectors(shots=-1)

    def test_invalid_text_type_rejected(self):
        with self.assertRaises(TypeError):
            symft.Circuit(text=object())

    def test_invalid_batch_size_rejected(self):
        circuit = symft.Circuit("M 0\n")

        with self.assertRaises(ValueError):
            circuit.compile_sampler(batch_size=-1)

    def test_circuit_cannot_be_reinitialized(self):
        circuit = symft.Circuit("M 0\n")

        with self.assertRaises(RuntimeError):
            circuit.__init__("NOT_A_GATE 0\n")

        self.assertEqual(circuit.num_qubits, 1)
        self.assertEqual(circuit.num_measurements, 1)


if __name__ == "__main__":
    unittest.main()
