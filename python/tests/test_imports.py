import unittest
import inspect

import symft


class ImportApiTest(unittest.TestCase):
    def test_module_exports_public_api(self):
        self.assertTrue(hasattr(symft, "Circuit"))
        self.assertTrue(hasattr(symft, "CompiledMeasurementSampler"))
        self.assertTrue(hasattr(symft, "CompiledCountsSampler"))
        self.assertTrue(hasattr(symft, "SymFTError"))
        self.assertTrue(hasattr(symft, "read_stim_file"))
        self.assertTrue(hasattr(symft, "sample"))
        self.assertTrue(hasattr(symft, "simd_backend"))
        self.assertTrue(hasattr(symft, "cuda_enabled"))
        self.assertTrue(hasattr(symft, "active_cuda_backend"))
        self.assertFalse(hasattr(symft, "active_simd_backend"))
        self.assertFalse(hasattr(symft, "active_batch_backend"))

    def test_backend_queries_return_strings(self):
        self.assertIsInstance(symft.simd_backend(), str)
        self.assertIsInstance(symft.active_cuda_backend(), str)
        self.assertIsInstance(symft.cuda_enabled(), bool)
        self.assertGreater(len(symft.simd_backend()), 0)
        self.assertGreater(len(symft.active_cuda_backend()), 0)

    def test_error_type_is_runtime_error(self):
        self.assertTrue(issubclass(symft.SymFTError, RuntimeError))

    def test_native_signatures_are_discoverable(self):
        self.assertEqual(str(inspect.signature(symft.Circuit)), "(text=None, path=None)")
        self.assertIn("shots=1", str(inspect.signature(symft.Circuit.sample)))
        self.assertIn("cuda=False", str(inspect.signature(symft.Circuit.sample_counts)))
        self.assertIn("cuda_mode='gpu'", str(inspect.signature(symft.Circuit.compile_counts_sampler)))
        self.assertIn("stream_id=None", str(inspect.signature(symft.CompiledCountsSampler.sample)))

    def test_version_is_exported(self):
        self.assertEqual(symft.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
