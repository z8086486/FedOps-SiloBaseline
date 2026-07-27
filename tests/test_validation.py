import unittest

from fedops_silo_baseline.validation import validate_baseline


class BaselineValidationTest(unittest.TestCase):
    def test_network_free_validation(self):
        result = validate_baseline(sample_count=16, max_batches=1)
        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "validate")
        self.assertEqual(result["input_shape"], [1, 28, 28])
        self.assertFalse(result["raw_data_uploaded"])


if __name__ == "__main__":
    unittest.main()
