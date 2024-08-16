import unittest
import torch
from three_d_scene_script.training_module import print_last_epoch_results
import io
from unittest.mock import patch

class TestPrintLastEpochResults(unittest.TestCase):
    def test_print_last_epoch_results(self):
        # Mock data
        last_epoch_predictions = [
            ("command1", torch.tensor([1.0, 2.0, 3.0])),
            ("command2", torch.tensor([4.0, 5.0, 6.0])),
            ("command3", torch.tensor([7.0, 8.0, 9.0]))
        ]
        last_epoch_ground_truths = [
            ("command1", torch.tensor([1.0, 2.0, 3.0])),
            ("command2", torch.tensor([4.0, 5.0, 6.0])),
            ("command3", torch.tensor([7.0, 8.0, 9.0]))
        ]

        # Capturing output by redirecting sys.stdout
        with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as fake_stdout:
            print_last_epoch_results(last_epoch_predictions, last_epoch_ground_truths)
            output = fake_stdout.getvalue()

        # Check for key strings in the output
        self.assertIn("Predictions and Ground Truths for the last epoch:", output)
        self.assertIn("Predicted Command: command1, Predicted Parameters: [1. 2. 3.]", output)
        self.assertIn("Ground Truth Command: command1, Ground Truth Parameters: [1. 2. 3.]", output)

if __name__ == '__main__':
    unittest.main()
