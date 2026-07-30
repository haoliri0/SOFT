import tempfile
import unittest
from pathlib import Path

import symft


class CircuitMetadataTest(unittest.TestCase):
    def test_empty_circuit_metadata_and_empty_samples(self):
        circuit = symft.Circuit("")

        self.assertEqual(circuit.num_qubits, 0)
        self.assertEqual(circuit.num_measurements, 0)
        self.assertEqual(circuit.num_detectors, 0)
        self.assertEqual(circuit.num_observables, 0)
        self.assertEqual(circuit.sample(shots=3).shape, (3, 0))

    def test_text_construction_metadata(self):
        circuit = symft.Circuit("X 0\nM 0\n")

        self.assertEqual(circuit.num_qubits, 1)
        self.assertEqual(circuit.num_measurements, 1)
        self.assertIn("qubits=1", repr(circuit))

    def test_file_construction_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "simple.stim"
            path.write_text("M 0\n", encoding="utf-8")

            circuit = symft.read_stim_file(str(path))

        self.assertEqual(circuit.num_qubits, 1)
        self.assertEqual(circuit.num_measurements, 1)

    def test_pathlike_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "simple.stim"
            path.write_text("M 0\n", encoding="utf-8")

            direct = symft.Circuit(path=path)
            helper = symft.read_stim_file(path)

        self.assertEqual(direct.num_measurements, 1)
        self.assertEqual(helper.num_measurements, 1)

    def test_detector_metadata(self):
        circuit = symft.Circuit("M 0\nDETECTOR(1.5, 2.5) rec[-1]\n")

        self.assertEqual(circuit.num_detectors, 1)
        self.assertEqual(circuit.detectors[0]["records"], (1,))
        self.assertEqual(circuit.detectors[0]["coords"], (1.5, 2.5))
        self.assertEqual(circuit.detectors[0]["line"], 2)

    def test_observable_metadata(self):
        circuit = symft.Circuit("M 0\nOBSERVABLE_INCLUDE(2) rec[-1]\n")

        self.assertEqual(circuit.num_observables, 3)
        self.assertEqual(circuit.num_observable_includes, 1)
        self.assertEqual(circuit.observables[0]["index"], 2)
        self.assertEqual(circuit.observables[0]["records"], (1,))

    def test_repeat_block_updates_record_count(self):
        circuit = symft.Circuit("REPEAT 3 {\nM 0\n}\n")

        self.assertEqual(circuit.num_measurements, 3)
        self.assertEqual(circuit.sample(shots=2).shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
