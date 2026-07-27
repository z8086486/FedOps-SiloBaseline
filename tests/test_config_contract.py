import unittest

from fedops_silo_baseline.config import load_config, validate_config


class ConfigContractTest(unittest.TestCase):
    def test_default_config_is_valid(self):
        config = load_config()
        self.assertEqual(config["dataset"]["name"], "MNIST")
        self.assertEqual(config["model"]["output_size"], 10)
        self.assertEqual(config["runtime_key"], "task_id")
        self.assertEqual(config["runtime"]["manager_port"], 8004)

    def test_invalid_batch_size_is_rejected(self):
        config = load_config()
        config["batch_size"] = 0
        with self.assertRaisesRegex(ValueError, "batch_size"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
