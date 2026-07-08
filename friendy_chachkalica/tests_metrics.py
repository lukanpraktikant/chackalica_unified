import unittest

import torch

from friendy_chachkalica.metrics import evaluate_detection


class EvaluateDetectionConfusionMatrixTests(unittest.TestCase):
    def test_confusion_matrix_uses_operating_threshold_not_map_floor(self):
        target = {
            "orig_size": torch.tensor([100, 100]),
            "boxes": torch.tensor([[40.0, 40.0, 60.0, 60.0]]),
            "labels": torch.tensor([0]),
        }
        prediction = torch.tensor([
            [0.5, 0.5, 0.2, 0.2, 0.9, 0.0],
            [0.1, 0.1, 0.1, 0.1, 0.2, 0.0],
        ])

        metrics = evaluate_detection(
            [prediction],
            [target],
            iou_thresholds=[0.5],
            map_score_threshold=0.1,
            score_threshold=0.5,
            num_classes=1,
        )

        self.assertEqual(metrics["num_predictions"], 1)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["confusion_matrix"]["conf_threshold"], 0.5)
        self.assertEqual(metrics["confusion_matrix"]["matrix"], [[1, 0], [0, 0]])


if __name__ == "__main__":
    unittest.main()
