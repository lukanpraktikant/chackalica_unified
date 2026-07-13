import unittest

import torch

from friendy_chachkalica.metrics import evaluate_detection, _matrix_from_confusion_data


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

    def test_confusion_matrix_data_rethresholds_interactively(self):
        """The stored event histogram rebuilds the matrix at any threshold >= floor.

        Two ground truths (class 0 and class 1). A high-confidence correct class-0
        detection, a mid-confidence class-0 prediction landing on the class-1 box
        (a confusion), and a low-confidence class-1 false positive. Raising the
        threshold should first drop the false positive, then turn the mid-confidence
        confusion into a miss.
        """
        target = {
            "orig_size": torch.tensor([100, 100]),
            "boxes": torch.tensor([[40.0, 40.0, 60.0, 60.0], [10.0, 10.0, 20.0, 20.0]]),
            "labels": torch.tensor([0, 1]),
        }
        prediction = torch.tensor([
            [0.50, 0.50, 0.20, 0.20, 0.90, 0.0],  # class0 @0.90 -> box0 correct
            [0.15, 0.15, 0.10, 0.10, 0.60, 0.0],  # class0 @0.60 -> box1 location, wrong class
            [0.90, 0.90, 0.05, 0.05, 0.20, 1.0],  # class1 @0.20 -> false positive
        ])

        metrics = evaluate_detection(
            [prediction],
            [target],
            iou_thresholds=[0.5],
            map_score_threshold=0.01,
            score_threshold=0.25,
            num_classes=2,
        )

        data = metrics["confusion_matrix_data"]
        self.assertEqual(data["floor"], 0.01)
        # Matrix rows: [truth cat, truth dog, background]; cols mirror + background.
        # At the operating threshold the stored matrix and a reconstruction agree.
        self.assertEqual(
            metrics["confusion_matrix"]["matrix"],
            _matrix_from_confusion_data(data, 0.25)["matrix"],
        )
        # Below every prediction: correct hit, the confusion, and the false positive.
        self.assertEqual(
            _matrix_from_confusion_data(data, 0.01)["matrix"],
            [[1, 0, 0], [1, 0, 0], [0, 1, 0]],
        )
        # Above the false positive (0.20) but below the confusion (0.60): FP gone.
        self.assertEqual(
            _matrix_from_confusion_data(data, 0.25)["matrix"],
            [[1, 0, 0], [1, 0, 0], [0, 0, 0]],
        )
        # Above the confusion too: its ground truth becomes a miss (background col).
        self.assertEqual(
            _matrix_from_confusion_data(data, 0.70)["matrix"],
            [[1, 0, 0], [0, 0, 1], [0, 0, 0]],
        )


if __name__ == "__main__":
    unittest.main()
