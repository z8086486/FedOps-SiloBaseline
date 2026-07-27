import unittest

import torch

from fedops_silo_baseline.data_preparation import (
    build_validation_loaders,
    describe_input_features,
)


class DataContractTest(unittest.TestCase):
    def test_input_contract_matches_mnist(self):
        contract = describe_input_features()
        self.assertEqual(contract["features"][0]["shape"], [1, 28, 28])
        self.assertEqual(len(contract["label"]["classes"]), 10)
        self.assertFalse(contract["raw_data_upload"])

    def test_synthetic_validation_data_is_deterministic(self):
        first_train, _ = build_validation_loaders(sample_count=16, batch_size=4, seed=7)
        second_train, _ = build_validation_loaders(sample_count=16, batch_size=4, seed=7)
        first_images, first_labels = next(iter(first_train))
        second_images, second_labels = next(iter(second_train))
        self.assertTrue(torch.equal(first_images, second_images))
        self.assertTrue(torch.equal(first_labels, second_labels))


if __name__ == "__main__":
    unittest.main()
