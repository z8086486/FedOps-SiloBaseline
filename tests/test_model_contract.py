import unittest

import torch

from fedops_silo_baseline.data_preparation import build_validation_loaders
from fedops_silo_baseline.model import build_model, evaluate_model, train_model


class ModelContractTest(unittest.TestCase):
    def test_model_output_and_short_training(self):
        model = build_model({"output_size": 10})
        output = model(torch.zeros(2, 1, 28, 28))
        self.assertEqual(list(output.shape), [2, 10])

        train_loader, validation_loader = build_validation_loaders(
            sample_count=16,
            batch_size=4,
            seed=42,
        )
        train_loss = train_model(
            model,
            train_loader,
            epochs=1,
            learning_rate=0.001,
            device=torch.device("cpu"),
            max_batches=1,
        )
        validation_loss, accuracy, metrics = evaluate_model(
            model,
            validation_loader,
            device=torch.device("cpu"),
            max_batches=1,
        )
        self.assertGreaterEqual(train_loss, 0)
        self.assertGreaterEqual(validation_loss, 0)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
        self.assertIn("f1_score", metrics)


if __name__ == "__main__":
    unittest.main()
